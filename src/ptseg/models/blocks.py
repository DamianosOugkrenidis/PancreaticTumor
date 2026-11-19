import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import as_tuple3, pad_to_multiple

class DropPath(nn.Module):

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}.")
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x.div(keep) * mask

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"

class Mlp3D(nn.Module):

    def __init__(self, in_features: int, hidden_features: int | None = None,
                 out_features: int | None = None, drop: float = 0.0,
                 act_layer: type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))

def get_norm(norm_name: str, channels: int) -> nn.Module:
    norm_name = (norm_name or "instance").lower()
    if norm_name in {"instance", "instance_norm", "in"}:
        return nn.InstanceNorm3d(channels, affine=True)
    if norm_name in {"batch", "batch_norm", "bn"}:
        return nn.BatchNorm3d(channels)
    if norm_name in {"group", "group_norm", "gn"}:
        groups = 8 if channels % 8 == 0 else 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unknown norm_name: {norm_name!r}")

def get_activation(name: str) -> nn.Module:
    name = (name or "leakyrelu").lower()
    if name in {"leakyrelu", "lrelu"}:
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name!r}")

class ConvNormAct3D(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int | None = None, norm_name: str = "instance",
                 act: str = "leakyrelu", bias: bool = True) -> None:
        padding = kernel_size // 2 if padding is None else padding
        layers: list[nn.Module] = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        ]
        if norm_name and norm_name.lower() not in {"none", "identity"}:
            layers.append(get_norm(norm_name, out_channels))
        if act and act.lower() not in {"none", "identity"}:
            layers.append(get_activation(act))
        super().__init__(*layers)

class ResBlock3D(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 norm_name: str = "instance", act: str = "leakyrelu") -> None:
        super().__init__()
        self.conv1 = ConvNormAct3D(in_channels, out_channels, 3, stride, 1, norm_name, act)
        self.conv2 = ConvNormAct3D(out_channels, out_channels, 3, 1, 1, norm_name, act="none")
        self.activation = get_activation(act)

        if in_channels != out_channels or stride != 1:
            self.shortcut = ConvNormAct3D(in_channels, out_channels, 1, stride, 0,
                                          norm_name, act="none", bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv2(self.conv1(x)) + self.shortcut(x))

class DeconvUpsample3D(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, factor: int = 2,
                 norm_name: str = "instance", act: str = "leakyrelu") -> None:
        if factor < 1 or (factor & (factor - 1)) != 0:
            raise ValueError(f"factor must be a power of two, got {factor}.")
        steps = int(round(math.log2(factor)))
        layers: list[nn.Module] = []
        current = in_channels
        for step in range(steps):
            target = out_channels if step == steps - 1 else max(out_channels, current // 2)
            layers.append(nn.ConvTranspose3d(current, target, kernel_size=2, stride=2))
            layers.append(get_norm(norm_name, target))
            layers.append(get_activation(act))
            current = target
        if steps == 0:
            layers.append(ConvNormAct3D(in_channels, out_channels, 3, 1, 1, norm_name, act))
        super().__init__(*layers)

class UpBlock3D(nn.Module):

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 norm_name: str = "instance", res_block: bool = True) -> None:
        super().__init__()
        self.up = DeconvUpsample3D(in_channels, out_channels, 2, norm_name)
        fuse = ResBlock3D(out_channels + skip_channels, out_channels, 1, norm_name) if res_block \
            else ConvNormAct3D(out_channels + skip_channels, out_channels, 3, 1, 1, norm_name)
        self.fuse = fuse

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-3:] != skip.shape[-3:]:
            raise ValueError(
                f"UpBlock3D: upsampled features {tuple(x.shape[-3:])} do not match skip "
                f"{tuple(skip.shape[-3:])}.")
        return self.fuse(torch.cat([x, skip], dim=1))

class UnetOutHead3D(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)

def sincos_position_embedding(grid_size: Sequence[int], dim: int) -> torch.Tensor:
    grid_size = as_tuple3(grid_size, "grid_size")
    if dim % 6 != 0:
        raise ValueError(f"dim must be divisible by 6 for a 3D sincos embedding, got {dim}.")
    per_axis = dim // 6
    omega = torch.arange(per_axis, dtype=torch.float32) / max(per_axis - 1, 1)
    omega = 1.0 / (10000 ** omega)

    embedding = torch.zeros((dim, *grid_size), dtype=torch.float32)
    for axis in range(3):
        position = torch.arange(grid_size[axis], dtype=torch.float32).unsqueeze(1)
        position = position * omega.unsqueeze(0)
        part = torch.cat([torch.sin(position), torch.cos(position)], dim=1)
        view_shape = [2 * per_axis if index == 0 else 1 for index in range(4)]
        view_shape[1 + axis] = grid_size[axis]
        part = part.reshape(view_shape).expand((2 * per_axis, *grid_size))
        embedding[axis * 2 * per_axis:(axis + 1) * 2 * per_axis] = part

    return embedding.flatten(1).transpose(0, 1).unsqueeze(0).contiguous()

class PatchEmbed3D(nn.Module):

    def __init__(self, in_channels: int, dim: int, patch_size: int = 16,
                 grid_size: Sequence[int] | None = None, pos_embed: str = "learnable",
                 proj_type: str = "conv", dropout: float = 0.0) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj_type = proj_type.lower()
        self.pos_embed_kind = (pos_embed or "none").lower()
        self.grid_size = as_tuple3(grid_size, "grid_size") if grid_size is not None else None

        if self.proj_type == "conv":
            self.proj = nn.Conv3d(in_channels, dim, kernel_size=self.patch_size,
                                  stride=self.patch_size)
        elif self.proj_type == "linear":
            self.proj = nn.Linear(in_channels * self.patch_size ** 3, dim)
        else:
            raise ValueError(f"Unknown proj_type: {proj_type!r}")

        if self.pos_embed_kind != "none":
            if self.grid_size is None:
                raise ValueError("pos_embed != none requires grid_size.")
            num_tokens = int(torch.tensor(self.grid_size).prod().item())
            if self.pos_embed_kind == "learnable":
                self.position = nn.Parameter(torch.zeros(1, num_tokens, dim))
                trunc_normal_(self.position)
            elif self.pos_embed_kind == "sincos":
                self.register_buffer("position", sincos_position_embedding(self.grid_size, dim),
                                     persistent=False)
            else:
                raise ValueError(f"Unknown pos_embed: {pos_embed!r}")
        else:
            self.position = None

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.proj_type == "conv":
            x = self.proj(x)
            tokens = x.flatten(2).transpose(1, 2)
        else:
            b, c, d, h, w = x.shape
            p = self.patch_size
            x = x.view(b, c, d // p, p, h // p, p, w // p, p)
            x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
            tokens = self.proj(x.view(b, -1, c * p ** 3))

        if self.position is not None:
            if self.position.shape[1] != tokens.shape[1]:
                raise ValueError(
                    f"Position embedding has {self.position.shape[1]} tokens but the input "
                    f"produces {tokens.shape[1]}. Check model.img_size vs data.roi_size.")
            tokens = tokens + self.position
        return self.drop(tokens)

def window_partition3d(x: torch.Tensor, window_size: int | Sequence[int]) -> torch.Tensor:
    wd, wh, ww = as_tuple3(window_size, "window_size")
    b, d, h, w, c = x.shape
    if d % wd or h % wh or w % ww:
        raise ValueError(f"Spatial size {(d, h, w)} is not divisible by window {(wd, wh, ww)}.")
    x = x.view(b, d // wd, wd, h // wh, wh, w // ww, ww, c)
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, wd * wh * ww, c)

def window_reverse3d(windows: torch.Tensor, window_size: int | Sequence[int],
                     dims: Sequence[int]) -> torch.Tensor:
    wd, wh, ww = as_tuple3(window_size, "window_size")
    d, h, w = int(dims[0]), int(dims[1]), int(dims[2])
    windows_per_sample = (d // wd) * (h // wh) * (w // ww)
    b = windows.shape[0] // windows_per_sample
    x = windows.view(b, d // wd, h // wh, w // ww, wd, wh, ww, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(b, d, h, w, -1)

class Attention3D(nn.Module):

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 attn_dropout: float = 0.0, proj_dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))

class WindowAttention3D(nn.Module):

    def __init__(self, dim: int, window_size: int | Sequence[int], num_heads: int,
                 qkv_bias: bool = True, attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.dim = dim
        self.window_size = as_tuple3(window_size, "window_size")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        wd, wh, ww = self.window_size
        table_size = (2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
        trunc_normal_(self.relative_position_bias_table)

        coords = torch.stack(torch.meshgrid(
            [torch.arange(size) for size in self.window_size], indexing="ij"))
        coords_flat = torch.flatten(coords, 1)
        relative = coords_flat[:, :, None] - coords_flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[..., 0] += wd - 1
        relative[..., 1] += wh - 1
        relative[..., 2] += ww - 1
        index = (relative[..., 0] * (2 * wh - 1) * (2 * ww - 1)
                 + relative[..., 1] * (2 * ww - 1)
                 + relative[..., 2])
        self.register_buffer("relative_position_index", index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)

        bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        bias = bias.reshape(n, n, -1).permute(2, 0, 1).contiguous().unsqueeze(0)
        attn = self.softmax(attn + bias)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))

class ViTBlock3D(nn.Module):

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 dropout: float = 0.0, attn_dropout: float = 0.0,
                 drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention3D(dim, num_heads, qkv_bias, attn_dropout, dropout)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp3D(dim, int(dim * mlp_ratio), drop=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))

class SwinBlock3D(nn.Module):

    def __init__(self, dim: int, num_heads: int, window_size: int | Sequence[int],
                 shift_size: int | Sequence[int] = 0, mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0, use_checkpoint: bool = False) -> None:
        super().__init__()
        window = as_tuple3(window_size, "window_size")
        shift = as_tuple3(shift_size, "shift_size")
        if any(s >= w for s, w in zip(shift, window)):
            raise ValueError(f"shift {shift} must be smaller than window {window} in every axis.")
        self.window_size = window
        self.shift_size = shift
        self.use_checkpoint = use_checkpoint

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = WindowAttention3D(dim, window, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp3D(dim, int(dim * mlp_ratio), drop=drop)

    def _shift_start(self, x: torch.Tensor, axis: int, amount: int) -> torch.Tensor:
        if amount <= 0:
            return x
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - axis)] = amount
        return F.pad(x, pad, mode="replicate")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, h, w, c = x.shape
        residual = x
        x = self.norm1(x)

        used_shift = tuple(min(self.shift_size[axis], max(0, x.shape[1 + axis] - 1))
                           for axis in range(3))
        if any(amount > 0 for amount in used_shift):
            xc = x.permute(0, 4, 1, 2, 3)
            for axis, amount in enumerate(used_shift):
                xc = self._shift_start(xc, axis, amount)
            x = xc.permute(0, 2, 3, 4, 1)
        shifted_dims = tuple(int(v) for v in x.shape[1:4])

        xc, _ = pad_to_multiple(x.permute(0, 4, 1, 2, 3), self.window_size, mode="constant")
        padded_dims = tuple(int(v) for v in xc.shape[-3:])
        x = xc.permute(0, 2, 3, 4, 1)

        windows = window_partition3d(x, self.window_size)
        if self.use_checkpoint and self.training:
            windows = torch.utils.checkpoint.checkpoint(self.attention, windows,
                                                        use_reentrant=False)
        else:
            windows = self.attention(windows)
        x = window_reverse3d(windows, self.window_size, padded_dims)
        x = x[:, :shifted_dims[0], :shifted_dims[1], :shifted_dims[2]]
        x = x[:, used_shift[0]:, used_shift[1]:, used_shift[2]:]
        x = x[:, :d, :h, :w]

        x = residual + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))

class PatchMerging3D(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(8 * dim, eps=1e-6)
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, h, w, c = x.shape
        if d % 2 or h % 2 or w % 2:
            raise ValueError(f"PatchMerging3D needs even spatial sizes, got {(d, h, w)}.")
        x = x.view(b, d // 2, 2, h // 2, 2, w // 2, 2, c)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(b, d // 2, h // 2, w // 2, 8 * c)
        return self.reduction(self.norm(x))

class PatchEmbedding3D(nn.Module):

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 2,
                 patch_stride: int = 2, norm_name: str = "instance") -> None:
        super().__init__()
        self.proj = ConvNormAct3D(in_channels, embed_dim, patch_size, patch_stride,
                                  padding=0, norm_name=norm_name, act="none", bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.permute(0, 2, 3, 4, 1).contiguous()

class OverlapPatchEmbedding3D(nn.Module):

    def __init__(self, in_channels: int, embed_dim: int, kernel_size: int,
                 stride: int, padding: int, norm_name: str = "instance") -> None:
        super().__init__()
        self.proj = ConvNormAct3D(in_channels, embed_dim, kernel_size, stride, padding,
                                  norm_name=norm_name, act="none", bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.permute(0, 2, 3, 4, 1).contiguous()

def confirm_divisible(value: int, divisor: int, name: str) -> None:
    if divisor <= 0 or value % divisor:
        raise ValueError(f"{name}={value} must be divisible by {divisor}.")
