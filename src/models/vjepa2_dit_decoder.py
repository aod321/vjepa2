import math
from typing import Optional

import torch
import torch.nn as nn


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings as in DiT/ADM."""

    if timesteps.ndim == 0:
        timesteps = timesteps[None]
    timesteps = timesteps.float()
    half = dim // 2
    device = timesteps.device
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, device=device).float() / half
    )
    args = timesteps[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, time_embed_dim: int) -> None:
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(timesteps, self.proj[0].in_features)
        return self.proj(emb)


class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_size: int, embed_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(embed_dim, hidden_size * 2)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(conditioning).chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + scale[:, None, :]) + shift[:, None, :]
        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        time_embed_dim: int,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.adaln1 = AdaLayerNorm(hidden_size, time_embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.adaln2 = AdaLayerNorm(hidden_size, time_embed_dim)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        h = self.adaln1(x, conditioning)
        attn_output, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.attn_dropout(attn_output)
        h = self.adaln2(x, conditioning)
        x = x + self.mlp(h)
        return x


class DiTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_dim: int, time_embed_dim: int) -> None:
        super().__init__()
        self.adaln = AdaLayerNorm(hidden_size, time_embed_dim)
        self.linear = nn.Linear(hidden_size, patch_dim)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        x = self.adaln(x, conditioning)
        return self.linear(x)


class VJEPA2DiTDecoder(nn.Module):
    """Diffusion-inspired transformer decoder for V-JEPA 2 tokens."""

    def __init__(
        self,
        encoder_dim: int,
        image_size: int = 256,
        patch_size: int = 16,
        tubelet_size: int = 2,
        channels: int = 3,
        decoder_embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        time_embed_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.channels = channels
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_size = image_size // patch_size
        self.spatial_tokens = self.grid_size * self.grid_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_embed_dim

        self.input_proj = (
            nn.Linear(encoder_dim, decoder_embed_dim) if encoder_dim != decoder_embed_dim else nn.Identity()
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.spatial_tokens, decoder_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        time_embed_dim = time_embed_dim or decoder_embed_dim * 4
        self.time_embed = TimestepEmbedder(decoder_embed_dim, time_embed_dim)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=decoder_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    time_embed_dim=time_embed_dim,
                    attention_dropout=attention_dropout,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        patch_dim = patch_size * patch_size * tubelet_size * channels
        self.final_layer = DiTFinalLayer(decoder_embed_dim, patch_dim, time_embed_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(tokens, (list, tuple)):
            tokens = torch.cat(tokens, dim=1)

        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens with 3 dims [B, N, D], got shape {tokens.shape}.")

        batch_size, num_tokens, dim = tokens.shape
        if dim != self.encoder_dim:
            raise ValueError(f"Token dim {dim} does not match encoder dim {self.encoder_dim}.")

        if num_tokens % self.spatial_tokens != 0:
            raise ValueError(
                f"Token count {num_tokens} not divisible by spatial tokens {self.spatial_tokens} per tubelet."
            )

        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        if timesteps.ndim == 1:
            if timesteps.shape[0] == 1 and batch_size > 1:
                timesteps = timesteps.expand(batch_size)
        if timesteps.shape[0] != batch_size:
            raise ValueError("timesteps must have same batch dimension as tokens")

        temporal_tokens = num_tokens // self.spatial_tokens

        if token_mask is not None:
            if token_mask.shape != (batch_size, num_tokens):
                raise ValueError("token_mask must match [B, N] shape of tokens.")
            token_mask = token_mask.view(batch_size, temporal_tokens, self.spatial_tokens)
            token_mask = token_mask.reshape(batch_size * temporal_tokens, self.spatial_tokens)

        frames = tokens.view(batch_size * temporal_tokens, self.spatial_tokens, dim)
        frames = self.input_proj(frames)

        pos_embed = self.pos_embed
        if pos_embed.dtype != frames.dtype:
            pos_embed = pos_embed.to(frames.dtype)
        frames = frames + pos_embed

        if token_mask is not None:
            frames = frames.masked_fill(token_mask.unsqueeze(-1), 0.0)

        frames = frames.view(batch_size, temporal_tokens, self.spatial_tokens, self.decoder_dim)
        frames = frames.reshape(batch_size, num_tokens, self.decoder_dim)

        time_cond = self.time_embed(timesteps)

        for block in self.blocks:
            frames = block(frames, time_cond)

        patches = self.final_layer(frames, time_cond)

        video = rearrange_patches(
            patches,
            batch_size=batch_size,
            temporal_tokens=temporal_tokens,
            grid_size=self.grid_size,
            patch_size=self.patch_size,
            tubelet_size=self.tubelet_size,
            channels=self.channels,
        )
        return video


class VJEPA2DiTDiffusionDecoder(nn.Module):
    """Transformer-based diffusion decoder that predicts noise in pixel space."""

    def __init__(
        self,
        encoder_dim: int,
        image_size: int = 256,
        patch_size: int = 16,
        tubelet_size: int = 2,
        channels: int = 3,
        decoder_embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        time_embed_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.channels = channels
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_size = image_size // patch_size
        self.spatial_tokens = self.grid_size * self.grid_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_embed_dim
        self.patch_dim = patch_size * patch_size * tubelet_size * channels

        self.patch_embed = nn.Linear(self.patch_dim, decoder_embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.spatial_tokens, decoder_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        time_embed_dim = time_embed_dim or decoder_embed_dim * 4
        self.time_embed = TimestepEmbedder(decoder_embed_dim, time_embed_dim)
        self.condition_proj = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, time_embed_dim),
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=decoder_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    time_embed_dim=time_embed_dim,
                    attention_dropout=attention_dropout,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = DiTFinalLayer(decoder_embed_dim, self.patch_dim, time_embed_dim)

    def forward(
        self,
        noisy_clip: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        encoder_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_clip.ndim != 5:
            raise ValueError("Expected noisy_clip [B, C, T, H, W].")

        batch_size, _, total_frames, height, width = noisy_clip.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Input height/width must be divisible by patch_size.")
        if total_frames % self.tubelet_size != 0:
            raise ValueError("Number of frames must be divisible by tubelet_size.")

        temporal_tokens = total_frames // self.tubelet_size
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        if grid_h != self.grid_size or grid_w != self.grid_size:
            raise ValueError("Input resolution must match decoder image_size.")

        patches = self._video_to_patches(noisy_clip, batch_size, temporal_tokens, grid_h, grid_w)
        tokens = self.patch_embed(patches)

        pos_embed = self.pos_embed
        if pos_embed.dtype != tokens.dtype:
            pos_embed = pos_embed.to(tokens.dtype)
        tokens = tokens + pos_embed

        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        if timesteps.ndim == 1 and timesteps.shape[0] == 1 and batch_size > 1:
            timesteps = timesteps.expand(batch_size)
        if timesteps.shape[0] != batch_size:
            raise ValueError("timesteps must have same batch dimension as input.")

        time_cond = self.time_embed(timesteps)
        cond_embed = encoder_tokens.mean(dim=1)
        cond_embed = self.condition_proj(cond_embed)
        conditioning = time_cond + cond_embed

        tokens = tokens.view(batch_size, temporal_tokens * self.spatial_tokens, self.decoder_dim)
        for block in self.blocks:
            tokens = block(tokens, conditioning)

        patch_noise = self.final_layer(tokens, conditioning)
        noise_video = self._patches_to_video(
            patch_noise,
            batch_size=batch_size,
            temporal_tokens=temporal_tokens,
            grid_size=self.grid_size,
        )
        return noise_video

    def _video_to_patches(
        self,
        clip: torch.Tensor,
        batch_size: int,
        temporal_tokens: int,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        clip = clip.view(
            batch_size,
            self.channels,
            temporal_tokens,
            self.tubelet_size,
            grid_h,
            self.patch_size,
            grid_w,
            self.patch_size,
        )
        clip = clip.permute(0, 2, 4, 6, 3, 5, 7, 1)
        patches = clip.reshape(
            batch_size * temporal_tokens,
            grid_h * grid_w,
            self.patch_dim,
        )
        return patches

    def _patches_to_video(
        self,
        patches: torch.Tensor,
        *,
        batch_size: int,
        temporal_tokens: int,
        grid_size: int,
    ) -> torch.Tensor:

        patches = patches.view(
            batch_size,
            temporal_tokens,
            grid_size,
            grid_size,
            self.patch_dim,
        )
        patches = patches.view(
            batch_size,
            temporal_tokens,
            grid_size,
            grid_size,
            self.tubelet_size,
            self.patch_size,
            self.patch_size,
            self.channels,
        )
        video = patches.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        video = video.view(
            batch_size,
            self.channels,
            temporal_tokens * self.tubelet_size,
            grid_size * self.patch_size,
            grid_size * self.patch_size,
        )
        return video


def rearrange_patches(
    patches: torch.Tensor,
    *,
    batch_size: int,
    temporal_tokens: int,
    grid_size: int,
    patch_size: int,
    tubelet_size: int,
    channels: int,
) -> torch.Tensor:
    return patches.view(batch_size, temporal_tokens, grid_size * grid_size, -1).reshape(
        batch_size,
        temporal_tokens,
        grid_size,
        grid_size,
        tubelet_size,
        patch_size,
        patch_size,
        channels,
    ).permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(
        batch_size,
        channels,
        temporal_tokens * tubelet_size,
        grid_size * patch_size,
        grid_size * patch_size,
    )
