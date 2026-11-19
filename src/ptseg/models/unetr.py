from typing import Sequence

import torch
import torch.nn as nn

from ..utils import as_tuple3
from .blocks import (DeconvUpsample3D, PatchEmbed3D, ResBlock3D, UnetOutHead3D, ViTBlock3D)

class ViTEncoder3D(nn.Module):

    def __init__(self, in_channels: int = 1, img_size: Sequence[int] = (96, 96, 96),
                 patch_size: int = 16, hidden_size: int = 768, mlp_dim: int = 3072,
                 num_layers: int = 12, num_heads: int = 12, proj_type: str = "conv",
                 pos_embed: str = "learnable", qkv_bias: bool = True, dropout: float = 0.0,
                 attn_dropout: float = 0.0, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        img_size = as_tuple3(img_size, "img_size")
        if any(size % patch_size for size in img_size):
            raise ValueError(f"img_size {img_size} must be divisible by patch_size={patch_size}.")
        if hidden_size % num_heads:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")

        self.img_size = img_size
        self.patch_size = int(patch_size)
        self.hidden_size = int(hidden_size)
        self.grid_size = tuple(size // patch_size for size in img_size)
        self.num_layers = int(num_layers)

        self.patch_embed = PatchEmbed3D(in_channels, hidden_size, patch_size, self.grid_size,
                                        pos_embed=pos_embed, proj_type=proj_type, dropout=dropout)
        mlp_ratio = mlp_dim / hidden_size
        drop_paths = torch.linspace(0, drop_path_rate, num_layers).tolist()
        self.blocks = nn.ModuleList([
            ViTBlock3D(hidden_size, num_heads, mlp_ratio, qkv_bias, dropout,
                       attn_dropout, drop_paths[layer]) for layer in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        tokens = self.patch_embed(x)
        hidden_states = [tokens]
        for block in self.blocks:
            tokens = block(tokens)
            hidden_states.append(tokens)
        return hidden_states

class TokensToVolume3D(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, grid_size: Sequence[int],
                 factor: int = 1, norm_name: str = "instance") -> None:
        super().__init__()
        self.grid_size = as_tuple3(grid_size, "grid_size")
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.up = DeconvUpsample3D(out_channels, out_channels, factor, norm_name)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, channels = tokens.shape
        if num_tokens != int(torch.tensor(self.grid_size).prod().item()):
            raise ValueError(
                f"Expected {int(torch.tensor(self.grid_size).prod().item())} tokens for grid "
                f"{self.grid_size} but received {num_tokens}. Check img_size vs the input size.")
        volume = tokens.transpose(1, 2).reshape(batch, channels, *self.grid_size)
        return self.up(self.proj(volume))

class UNETR3D(nn.Module):

    def __init__(self, in_channels: int = 1, out_channels: int = 3,
                 img_size: Sequence[int] = (96, 96, 96), patch_size: int = 16,
                 feature_size: int = 16, hidden_size: int = 768, mlp_dim: int = 3072,
                 num_heads: int = 12, num_layers: int = 12, proj_type: str = "conv",
                 pos_embed: str = "learnable", qkv_bias: bool = True, dropout: float = 0.0,
                 attn_dropout: float = 0.0, drop_path_rate: float = 0.1,
                 res_block: bool = True, norm_name: str = "instance",
                 skip_indices: Sequence[int] = (3, 6, 9, 12),
                 decoder_channels: Sequence[int] = (256, 128, 64, 16)) -> None:
        super().__init__()
        skip_indices = tuple(int(index) for index in skip_indices)
        decoder_channels = tuple(int(channel) for channel in decoder_channels)
        if len(skip_indices) != 4:
            raise ValueError("skip_indices must contain exactly four transformer layers.")
        if len(decoder_channels) != 4:
            raise ValueError("decoder_channels must contain exactly four widths.")
        if max(skip_indices) > num_layers:
            raise ValueError(f"skip_indices {skip_indices} exceed num_layers={num_layers}.")
        if decoder_channels[-1] != feature_size:
            raise ValueError(
                f"The last decoder width ({decoder_channels[-1]}) must equal feature_size "
                f"({feature_size}) because it feeds the output head.")

        self.skip_indices = skip_indices
        self.encoder = ViTEncoder3D(in_channels, img_size, patch_size, hidden_size, mlp_dim,
                                    num_layers, num_heads, proj_type, pos_embed, qkv_bias,
                                    dropout, attn_dropout, drop_path_rate)
        grid_size = self.encoder.grid_size

        self.input_skip = ResBlock3D(in_channels, feature_size, 1, norm_name)

        self.skip9 = TokensToVolume3D(hidden_size, decoder_channels[1], grid_size, 2, norm_name)
        self.skip6 = TokensToVolume3D(hidden_size, decoder_channels[1], grid_size, 4, norm_name)
        self.skip3 = TokensToVolume3D(hidden_size, decoder_channels[2], grid_size, 8, norm_name)

        self.stage0 = TokensToVolume3D(hidden_size, decoder_channels[0], grid_size, 2, norm_name)
        self.fuse0 = ResBlock3D(decoder_channels[0] + decoder_channels[1],
                                decoder_channels[0], 1, norm_name) if res_block else \
            DeconvUpsample3D(decoder_channels[0] + decoder_channels[1], decoder_channels[0], 1,
                             norm_name, act="none")

        self.up1 = DeconvUpsample3D(decoder_channels[0], decoder_channels[1], 2, norm_name)
        self.fuse1 = ResBlock3D(decoder_channels[1] * 2, decoder_channels[1], 1, norm_name)

        self.up2 = DeconvUpsample3D(decoder_channels[1], decoder_channels[2], 2, norm_name)
        self.fuse2 = ResBlock3D(decoder_channels[2] * 2, decoder_channels[2], 1, norm_name)

        self.up3 = DeconvUpsample3D(decoder_channels[2], decoder_channels[3], 2, norm_name)
        self.fuse3 = ResBlock3D(decoder_channels[3] + feature_size, decoder_channels[3], 1, norm_name)

        self.head = UnetOutHead3D(feature_size, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"UNETR3D expects (B, C, D, H, W) input, got {tuple(x.shape)}.")

        hidden = self.encoder(x)
        z3 = hidden[self.skip_indices[0]]
        z6 = hidden[self.skip_indices[1]]
        z9 = hidden[self.skip_indices[2]]
        z12 = hidden[self.skip_indices[3]]

        decoded = self.stage0(z12)
        decoded = self.fuse0(torch.cat([decoded, self.skip9(z9)], dim=1))

        decoded = self.up1(decoded)
        decoded = self.fuse1(torch.cat([decoded, self.skip6(z6)], dim=1))

        decoded = self.up2(decoded)
        decoded = self.fuse2(torch.cat([decoded, self.skip3(z3)], dim=1))

        decoded = self.up3(decoded)
        decoded = self.fuse3(torch.cat([decoded, self.input_skip(x)], dim=1))

        return self.head(decoded)

    def describe(self) -> dict:
        return {
            "name": "unetr",
            "grid_size": self.encoder.grid_size,
            "tokens": int(torch.tensor(self.encoder.grid_size).prod().item()),
            "hidden_size": self.encoder.hidden_size,
            "skip_indices": self.skip_indices,
        }
