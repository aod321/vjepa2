import torch
import torch.nn as nn


class VisionTransformerDecoder(nn.Module):
    """Inverse of Vision Transformer patch embedding."""

    def __init__(self, encoder_vit_model, channel_size=3):
        super().__init__()
        self.vit_model = encoder_vit_model
        self.channel_size = channel_size
        self.patch_size = getattr(encoder_vit_model, "patch_size", 16)
        self.tubelet_size = getattr(encoder_vit_model, "tubelet_size", 1)
        self.embed_dim = getattr(encoder_vit_model, "embed_dim")
        self.is_video = getattr(encoder_vit_model, "is_video", False)

        if self.is_video:
            self.decoder = nn.ConvTranspose3d(
                in_channels=self.embed_dim,
                out_channels=channel_size,
                kernel_size=(self.tubelet_size, self.patch_size, self.patch_size),
                stride=(self.tubelet_size, self.patch_size, self.patch_size),
            )
        else:
            self.decoder = nn.ConvTranspose2d(
                in_channels=self.embed_dim,
                out_channels=channel_size,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )

    def forward(self, h):
        if not torch.is_tensor(h):
            h = torch.stack(h, dim=0)

        B, N, D = h.shape
        if D != self.embed_dim:
            raise ValueError(f"Token dim {D} does not match ViT embed dim {self.embed_dim}.")

        if self.is_video:
            return self._decode_video_tokens(h, B, N)
        return self._decode_image_tokens(h, B, N)

    def _decode_image_tokens(self, h, batch_size, num_tokens):
        img_height = self.vit_model.img_height
        img_width = self.vit_model.img_width
        patches_per_row = img_height // self.patch_size
        patches_per_col = img_width // self.patch_size
        expected_tokens = patches_per_row * patches_per_col

        if num_tokens == expected_tokens + 1:
            h = h[:, 1:, :]
            num_tokens -= 1

        if num_tokens != expected_tokens:
            raise ValueError(
                f"Token count {num_tokens} does not match expected grid {patches_per_row}x{patches_per_col}."
            )

        h = h.transpose(1, 2).reshape(batch_size, self.embed_dim, patches_per_row, patches_per_col)
        return self.decoder(h)

    def _decode_video_tokens(self, h, batch_size, num_tokens):
        img_height = self.vit_model.img_height
        img_width = self.vit_model.img_width

        patches_per_row = img_height // self.patch_size
        patches_per_col = img_width // self.patch_size
        patches_per_frame = patches_per_row * patches_per_col

        if num_tokens % patches_per_frame != 0:
            raise ValueError(
                f"Token count {num_tokens} not divisible by patches per frame {patches_per_frame}."
            )

        temporal_patches = num_tokens // patches_per_frame
        h = h.transpose(1, 2).reshape(
            batch_size, self.embed_dim, temporal_patches, patches_per_row, patches_per_col
        )
        return self.decoder(h)
