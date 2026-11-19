from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import as_tuple3
from .blocks import DropPath, OverlapPatchEmbedding3D, get_norm

class EfficientSelfAttention3D(nn.Module):

    def __init__(self, dim: int, num_heads: int, sr_ratio: int = 1, qkv_bias: bool = True,
                 attn_dropout: float = 0.0, proj_dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = int(sr_ratio)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

        if self.sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.norm = nn.LayerNorm(dim, eps=1e-6)

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.reshape(batch, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, depth, height, width, channels = x.shape

        query = self._to_heads(self.q(x).reshape(batch, depth * height * width, channels))

        if self.sr_ratio > 1:
            reduced = x.permute(0, 4, 1, 2, 3)
            reduced = self.sr(reduced)
            reduced = reduced.permute(0, 2, 3, 4, 1).reshape(batch, -1, channels)
            reduced = self.norm(reduced)
        else:
            reduced = x.reshape(batch, depth * height * width, channels)

        key = self._to_heads(self.k(reduced))
        value = self._to_heads(self.v(reduced))

        attention = ((query * self.scale) @ key.transpose(-2, -1)).softmax(dim=-1)
        attention = self.attn_drop(attention)

        out = (attention @ value).transpose(1, 2).reshape(batch, depth, height, width, channels)
        return self.proj_drop(self.proj(out))

class MixFFN3D(nn.Module):

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                                groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.act(self.dwconv(x))
        x = x.permute(0, 2, 3, 4, 1)
        return self.drop(self.fc2(x))

class MixTransformerBlock3D(nn.Module):

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, sr_ratio: int = 1,
                 qkv_bias: bool = True, dropout: float = 0.0, attn_dropout: float = 0.0,
                 drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = EfficientSelfAttention3D(dim, num_heads, sr_ratio, qkv_bias,
                                             attn_dropout, dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = MixFFN3D(dim, int(dim * mlp_ratio), dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.ffn(self.norm2(x)))

class MiTStage3D(nn.Module):

    def __init__(self, in_channels: int, embed_dim: int, depth: int, num_heads: int,
                 sr_ratio: int, patch_size: int, patch_stride: int, patch_padding: int,
                 mlp_ratio: float, qkv_bias: bool, dropout: float, attn_dropout: float,
                 drop_paths: Sequence[float], norm_name: str) -> None:
        super().__init__()
        self.patch_embed = OverlapPatchEmbedding3D(in_channels, embed_dim, patch_size,
                                                   patch_stride, patch_padding, norm_name)
        self.blocks = nn.ModuleList([
            MixTransformerBlock3D(embed_dim, num_heads, mlp_ratio, sr_ratio, qkv_bias,
                                  dropout, attn_dropout, drop_paths[index])
            for index in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        return x

class SegFormer3D(nn.Module):

    def __init__(self, in_channels: int = 1, out_channels: int = 3,
                 img_size: Sequence[int] = (96, 96, 96),
                 embed_dims: Sequence[int] = (32, 64, 160, 256),
                 depths: Sequence[int] = (2, 2, 2, 2), num_heads: Sequence[int] = (1, 2, 5, 8),
                 sr_ratios: Sequence[int] = (8, 4, 2, 1),
                 patch_sizes: Sequence[int] = (7, 3, 3, 3),
                 patch_strides: Sequence[int] = (4, 2, 2, 2),
                 patch_paddings: Sequence[int] = (3, 1, 1, 1), mlp_ratio: float = 4.0,
                 qkv_bias: bool = True, drop_rate: float = 0.0, attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.1, decoder_dim: int = 256,
                 norm_name: str = "instance") -> None:
        super().__init__()
        self.img_size = as_tuple3(img_size, "img_size")
        parsed = [tuple(int(v) for v in values) for values in
                  (embed_dims, depths, num_heads, sr_ratios, patch_sizes, patch_strides,
                   patch_paddings)]
        (embed_dims, depths, num_heads, sr_ratios, patch_sizes, patch_strides,
         patch_paddings) = parsed
        num_stages = len(embed_dims)
        for name, values in zip(("depths", "num_heads", "sr_ratios", "patch_sizes",
                                 "patch_strides", "patch_paddings"), parsed[1:]):
            if len(values) != num_stages:
                raise ValueError(f"{name} must have the same length as embed_dims.")
        self.num_stages = num_stages
        self.embed_dims = list(embed_dims)

        total_depth = sum(depths)
        drop_paths = torch.linspace(0, drop_path_rate, total_depth).tolist()

        encoder_stages = []
        offset = 0
        stage_in = in_channels
        for stage in range(num_stages):
            encoder_stages.append(MiTStage3D(
                in_channels=stage_in, embed_dim=embed_dims[stage], depth=depths[stage],
                num_heads=num_heads[stage], sr_ratio=sr_ratios[stage],
                patch_size=patch_sizes[stage], patch_stride=patch_strides[stage],
                patch_padding=patch_paddings[stage], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                dropout=drop_rate, attn_dropout=attn_drop_rate,
                drop_paths=drop_paths[offset:offset + depths[stage]], norm_name=norm_name))
            offset += depths[stage]
            stage_in = embed_dims[stage]
        self.encoder = nn.ModuleList(encoder_stages)

        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(self.embed_dims[stage], decoder_dim),
                          nn.ReLU(inplace=True)) for stage in range(num_stages)
        ])
        self.fuse = nn.Conv3d(decoder_dim * num_stages, decoder_dim, kernel_size=1)
        self.fuse_norm = get_norm(norm_name, decoder_dim)
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Conv3d(decoder_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"SegFormer3D expects (B, C, D, H, W) input, got {tuple(x.shape)}.")
        input_size = x.shape[-3:]

        features = []
        current = x
        for stage in self.encoder:
            out = stage(current)
            features.append(out)
            current = out.permute(0, 4, 1, 2, 3).contiguous()

        base_size = features[0].shape[1:4]
        fused = []
        for feature, projection in zip(features, self.projections):
            projected = projection(feature).permute(0, 4, 1, 2, 3).contiguous()
            if projected.shape[-3:] != base_size:
                projected = F.interpolate(projected, size=base_size, mode="trilinear",
                                          align_corners=False)
            fused.append(projected)

        x = self.dropout(self.fuse_norm(self.fuse(torch.cat(fused, dim=1))))
        logits = self.head(x)
        return F.interpolate(logits, size=input_size, mode="trilinear", align_corners=False)

    def describe(self) -> dict:
        return {
            "name": "segformer3d",
            "embed_dims": self.embed_dims,
            "num_stages": self.num_stages,
            "decoder_dim": self.fuse.out_channels,
        }
