import torch
import torch.nn as nn

try:
    from diffusers import UNetSpatioTemporalConditionModel
except ImportError as exc:  # pragma: no cover - library dependency gate
    raise ImportError(
        "Installing `diffusers` is required for VJEPA2DiffusionDecoder. "
        "Please install diffusers>=0.19 to enable the diffusion decoder."
    ) from exc


class VJEPA2DiffusionDecoder(nn.Module):
    """
    Conditional diffusion decoder that predicts noise on video clips given encoder tokens.

    Args mirror the standard V-JEPA decoder where applicable. The underlying network is a
    `UNetSpatioTemporalConditionModel` from HuggingFace diffusers conditioned on encoder tokens
    through cross-attention. The module outputs predicted noise with the same shape as the noisy
    input clip. Training should follow a DDPM-style objective outside this module.
    """

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
        time_embed_dim: int | None = None,
        num_frames: int | None = None,
        unet_config: dict | None = None,
    ) -> None:
        super().__init__()

        unet_config = unet_config or {}
        _ = patch_size  # Unused but kept for interface parity

        # Map encoder tokens to cross-attention dimension.
        self.condition_proj = nn.Linear(encoder_dim, decoder_embed_dim)
        self.channels = channels
        self.image_size = image_size
        self.tubelet_size = tubelet_size

        default_block_out_channels = (128, 256, 256, 256)
        block_out_channels = tuple(unet_config.get("block_out_channels", default_block_out_channels))
        num_blocks = len(block_out_channels)

        down_block_types_default = (
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        )
        down_block_types = tuple(unet_config.get("down_block_types", down_block_types_default))
        up_block_types_default = (
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        )
        up_block_types = tuple(unet_config.get("up_block_types", up_block_types_default))

        if len(down_block_types) != len(up_block_types):
            raise ValueError("down_block_types and up_block_types must have the same length.")
        if len(block_out_channels) != len(down_block_types):
            raise ValueError("block_out_channels must align with block types.")

        default_heads = tuple(max(1, num_heads // (2 ** i)) for i in range(num_blocks))
        attention_heads = unet_config.get("num_attention_heads", default_heads)
        if isinstance(attention_heads, int):
            attention_heads = (attention_heads,) * num_blocks
        attention_heads = tuple(attention_heads)
        if len(attention_heads) != num_blocks:
            raise ValueError("num_attention_heads must match number of blocks.")

        default_layers_per_block = unet_config.get("layers_per_block", 1)
        if isinstance(default_layers_per_block, int):
            layers_per_block_list = [default_layers_per_block] * num_blocks
        else:
            layers_per_block_list = list(default_layers_per_block)
        if len(layers_per_block_list) != num_blocks:
            raise ValueError("layers_per_block must match number of blocks.")
        layers_per_block = tuple(layers_per_block_list)

        addition_time_embed_dim = unet_config.get("addition_time_embed_dim", time_embed_dim or max(decoder_embed_dim // 4, 1))
        if decoder_embed_dim % addition_time_embed_dim != 0:
            addition_time_embed_dim = decoder_embed_dim
        self._add_embedding_components = decoder_embed_dim // addition_time_embed_dim

        self.unet = UNetSpatioTemporalConditionModel(
            sample_size=image_size,
            in_channels=channels,
            out_channels=channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            addition_time_embed_dim=addition_time_embed_dim,
            projection_class_embeddings_input_dim=decoder_embed_dim,
            layers_per_block=layers_per_block,
            cross_attention_dim=decoder_embed_dim,
            transformer_layers_per_block=1,
            num_attention_heads=attention_heads,
            num_frames=num_frames or tubelet_size * 8,
        )

    def forward(
        self,
        noisy_clip: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        encoder_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict diffusion noise for noisy clips conditioned on encoder tokens.

        Args:
            noisy_clip: Tensor of shape [B, C, T, H, W] containing noisy video frames.
            timesteps: Tensor of shape [B] holding diffusion time steps.
            encoder_tokens: Tensor [B, N, D_enc] with encoder outputs.
        """
        if encoder_tokens.ndim != 3:
            raise ValueError("encoder_tokens must be [B, N, D].")

        if noisy_clip.ndim != 5:
            raise ValueError("noisy_clip must be [B, C, T, H, W].")

        batch_size, _, frames, height, width = noisy_clip.shape
        sample = noisy_clip.permute(0, 2, 1, 3, 4).contiguous()

        cond = self.condition_proj(encoder_tokens)

        components = []
        if self._add_embedding_components >= 1:
            components.append(float(self.image_size))
        if self._add_embedding_components >= 2:
            components.append(float(height))
        if self._add_embedding_components >= 3:
            components.append(float(frames))
        if self._add_embedding_components >= 4:
            components.append(float(self.tubelet_size))
        while len(components) < self._add_embedding_components:
            components.append(0.0)

        added_time_ids = sample.new_tensor(components).view(1, -1).repeat(batch_size, 1)

        return self.unet(
            sample=sample,
            timestep=timesteps,
            encoder_hidden_states=cond,
            added_time_ids=added_time_ids,
        ).sample.permute(0, 2, 1, 3, 4).contiguous()
