from typing import Sequence

import torch
import torch.nn as nn

from ..utils import as_tuple3
from .blocks import (DeconvUpsample3D, PatchEmbedding3D, PatchMerging3D, ResBlock3D, SwinBlock3D,
                     UnetOutHead3D)

class SwinStage3D(nn.Module):

    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int,
                 shift_size: int, mlp_ratio: float, qkv_bias: bool, drop: float,
                 attn_drop: float, drop_paths: Sequence[float], use_checkpoint: bool,
                 downsample: bool) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock3D(dim, num_heads, window_size,
                        shift_size=shift_size if index % 2 else 0,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                        drop_path=drop_paths[index], use_checkpoint=use_checkpoint)
            for index in range(depth)
        ])
        self.downsample = PatchMerging3D(dim) if downsample else None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        for block in self.blocks:
            x = block(x)
        merged = self.downsample(x) if self.downsample is not None else None
        return x, merged

class SwinUNETR3D(nn.Module):

    def __init__(self, in_channels: int = 1, out_channels: int = 3,
                 img_size: Sequence[int] = (96, 96, 96), patch_size: int = 2,
                 feature_size: int = 48, depths: Sequence[int] = (2, 2, 2, 2),
                 num_heads: Sequence[int] = (3, 6, 12, 24), window_size: int = 7,
                 window_shift: int = 3, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop_rate: float = 0.0, attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.1, res_block: bool = True,
                 norm_name: str = "instance", use_checkpoint: bool = False) -> None:
        super().__init__()
        depths = tuple(int(depth) for depth in depths)
        num_heads = tuple(int(head) for head in num_heads)
        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have the same length.")
        if len(depths) < 2:
            raise ValueError("Swin UNETR needs at least two stages.")
        if len(depths) != 4:
            raise ValueError("This Swin UNETR implementation expects exactly four stages "
                             "(decoder paths are wired for 4 levels).")

        img_size = as_tuple3(img_size, "img_size")
        self.img_size = img_size
        self.patch_size = int(patch_size)
        self.num_stages = len(depths)
        self.dims = [int(feature_size) * 2 ** stage for stage in range(self.num_stages)]

        stride = self.patch_size * 2 ** (self.num_stages - 1)
        if any(size % stride for size in img_size):
            raise ValueError(
                f"img_size {img_size} must be divisible by {stride} "
                f"(patch_size * 2**{self.num_stages - 1}).")

        self.patch_embed = PatchEmbedding3D(in_channels, self.dims[0], self.patch_size,
                                            self.patch_size, norm_name)

        total_depth = sum(depths)
        drop_paths = torch.linspace(0, drop_path_rate, total_depth).tolist()
        offset = 0
        stages = []
        for stage, depth in enumerate(depths):
            stages.append(SwinStage3D(
                dim=self.dims[stage], depth=depth, num_heads=num_heads[stage],
                window_size=window_size, shift_size=window_shift, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_paths=drop_paths[offset:offset + depth], use_checkpoint=use_checkpoint,
                downsample=stage < self.num_stages - 1))
            offset += depth
        self.stages = nn.ModuleList(stages)

        self.up3 = DeconvUpsample3D(self.dims[3], self.dims[2], 2, norm_name)
        self.fuse3 = ResBlock3D(self.dims[2] * 2, self.dims[2], 1, norm_name)

        self.up2 = DeconvUpsample3D(self.dims[2], self.dims[1], 2, norm_name)
        self.fuse2 = ResBlock3D(self.dims[1] * 2, self.dims[1], 1, norm_name)

        self.up1 = DeconvUpsample3D(self.dims[1], self.dims[0], 2, norm_name)
        self.fuse1 = ResBlock3D(self.dims[0] * 2, self.dims[0], 1, norm_name)

        self.up_full = DeconvUpsample3D(self.dims[0], self.dims[0], 2, norm_name)
        self.input_skip = ResBlock3D(in_channels, self.dims[0], 1, norm_name)
        self.fuse_full = ResBlock3D(self.dims[0] * 2, self.dims[0], 1, norm_name)

        self.head = UnetOutHead3D(self.dims[0], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"SwinUNETR3D expects (B, C, D, H, W) input, got {tuple(x.shape)}.")

        features = [self.patch_embed(x)]
        tokens = features[0]
        for stage in self.stages:
            stage_features, tokens = stage(tokens)
            features.append(stage_features)

        deepest, skip_h8, skip_h4, skip_h2 = features[-1], features[-2], features[-3], features[-4]

        decoded = self.fuse3(self._concat(self.up3(self._to_first_dim(deepest)),
                                          self._to_first_dim(skip_h8)))
        decoded = self.fuse2(self._concat(self.up2(decoded), self._to_first_dim(skip_h4)))
        decoded = self.fuse1(self._concat(self.up1(decoded), self._to_first_dim(skip_h2)))
        decoded = self.fuse_full(self._concat(self.up_full(decoded), self.input_skip(x)))
        return self.head(decoded)

    @staticmethod
    def _to_first_dim(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 4, 1, 2, 3).contiguous()

    @staticmethod
    def _concat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.shape[-3:] != b.shape[-3:]:
            raise ValueError(
                f"Swin UNETR decoder mismatch: {tuple(a.shape[-3:])} vs {tuple(b.shape[-3:])}.")
        return torch.cat([a, b], dim=1)

    def describe(self) -> dict:
        return {
            "name": "swin_unetr",
            "dims": self.dims,
            "num_stages": self.num_stages,
            "downsample_stride": self.patch_size * 2 ** (self.num_stages - 1),
        }
