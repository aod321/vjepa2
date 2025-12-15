from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

class FrameRefinementHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.act = nn.GELU()
        self.enc1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.enc2 = nn.Conv3d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=(1, 2, 2),
            padding=1,
        )
        self.dec1 = nn.ConvTranspose3d(
            hidden_channels,
            hidden_channels,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )
        self.dec2 = nn.Conv3d(hidden_channels, in_channels, kernel_size=3, padding=1)

        nn.init.zeros_(self.dec2.weight)
        nn.init.zeros_(self.dec2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.enc1(x))
        down = self.act(self.enc2(y))
        up = self.act(self.dec1(down))
        up = up + y
        residual = self.dec2(self.act(up))
        return x + residual


class VJEPA2FrameDecoder(nn.Module):
    """
    Frame decoder that mirrors ViT-L capacity to map encoder tokens back to pixels.

    Args:
        encoder_dim: Dimension of encoder tokens (`D`).
        image_size: Spatial resolution of reconstructed frames (assumes square inputs).
        patch_size: Spatial patch size used by the encoder.
        tubelet_size: Temporal patch size used by the encoder (defaults to 1 for images).
        channels: Number of output channels to reconstruct (e.g. 3 for RGB).
        decoder_embed_dim: Hidden size for the decoder transformer (ViT-L default: 1024).
        depth: Number of transformer layers (ViT-L default: 24).
        num_heads: Number of attention heads (ViT-L default: 16).
        mlp_ratio: Width multiplier for the MLP inside each transformer block.
        dropout: Dropout applied inside transformer layers.
        attention_dropout: Dropout applied to attention weights.
    """

    def __init__(
        self,
        encoder_dim: int,
        image_size: int = 256,
        patch_size: int = 16,
        tubelet_size: int = 1,
        channels: int = 3,
        decoder_embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.channels = channels
        self.image_size = image_size
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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=num_heads,
            dim_feedforward=int(decoder_embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        encoder_layer.self_attn.dropout = attention_dropout
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.patch_head = nn.Linear(
            decoder_embed_dim, patch_size * patch_size * tubelet_size * channels
        )
        refine_hidden = min(decoder_embed_dim // 4, 256)
        self.refine_head = FrameRefinementHead(channels, refine_hidden)

    def forward(self, tokens, *, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tokens: Tensor of shape [B, N, D_enc] with encoder outputs.
            token_mask: Optional boolean mask of shape [B, N]; masked tokens are ignored.

        Returns:
            Reconstructed clip with shape [B, C, T, H, W].
        """
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

        temporal_tokens = num_tokens // self.spatial_tokens
        if token_mask is not None:
            if token_mask.shape != (batch_size, num_tokens):
                raise ValueError("token_mask must match [B, N] shape of tokens.")
            # expand mask per temporal slice
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

        frames = self.transformer(frames)
        frames = self.norm(frames)

        patches = self.patch_head(frames)

        video = rearrange(
            patches,
            "(b t) (gh gw) (pf ph pw c) -> b c (t pf) (gh ph) (gw pw)",
            b=batch_size,
            t=temporal_tokens,
            gh=self.grid_size,
            gw=self.grid_size,
            pf=self.tubelet_size,
            ph=self.patch_size,
            pw=self.patch_size,
            c=self.channels,
        )
        video = self.refine_head(video)
        return video
