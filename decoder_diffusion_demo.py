#%%
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from diffusers import DDPMScheduler
from torchvision.io import read_video
from torchvision.transforms.functional import InterpolationMode, resize

import src.models.vision_transformer as video_vit
from src.models.vjepa2_dit_decoder import VJEPA2DiTDiffusionDecoder

import matplotlib.pyplot as plt


#%%
CHECKPOINT_PATH = "/nvmessd/yinzi/vjepa2/checkpoints/decoder_diffusion/latest.pt"
DATASET_ROOT = Path("/nvmessd/yinzi/vjepa2/datasets/go_stanford_converted")
CSV_PATH = DATASET_ROOT / "go_stanford_train_paths.csv"

IMAGE_MEAN_255 = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1, 1) * 255.0
IMAGE_STD_255 = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1, 1) * 255.0

NUM_FRAMES = 8
NUM_INFERENCE_STEPS = 50
DECODER_EMBED_DIM = 384
DECODER_DEPTH = 8
DECODER_NUM_HEADS = 12
TIME_EMBED_DIM = 768


#%%
def load_checkpoint(path: str):
    checkpoint = torch.load(path, map_location="cpu")
    decoder_state = checkpoint["decoder"]
    encoder_state = checkpoint["encoder"]
    return decoder_state, encoder_state


def prepare_clip(num_frames: int = NUM_FRAMES):
    with open(CSV_PATH, "r") as f:
        sample_dir = Path(f.readline().strip())

    video_path = sample_dir / "recordings" / "MP4" / "nav_camera.mp4"
    print(f"Using video: {video_path}")
    frames, _, _ = read_video(str(video_path), pts_unit="sec")
    if frames.size(0) < num_frames:
        raise RuntimeError(f"Video {video_path} shorter than {num_frames} frames.")

    clip = frames[:num_frames].to(torch.float32)  # [T, H, W, C]
    resized = [
        resize(
            frame.permute(2, 0, 1),
            [256, 256],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        for frame in clip
    ]
    clip_tensor = torch.stack(resized, dim=0)  # [T, C, H, W]
    clip_tensor = clip_tensor.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]
    return clip_tensor


#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder_state, encoder_state = load_checkpoint(CHECKPOINT_PATH)

decoder_model = VJEPA2DiTDiffusionDecoder(
    encoder_dim=1024,
    image_size=256,
    patch_size=16,
    tubelet_size=2,
    channels=3,
    decoder_embed_dim=DECODER_EMBED_DIM,
    depth=DECODER_DEPTH,
    num_heads=DECODER_NUM_HEADS,
    mlp_ratio=4.0,
    time_embed_dim=TIME_EMBED_DIM,
)
decoder_model.load_state_dict(decoder_state)
decoder_model = decoder_model.to(device).eval()

encoder_model = video_vit.vit_large(
    patch_size=16,
    img_size=256,
    num_frames=NUM_FRAMES,
    tubelet_size=2,
    uniform_power=True,
    use_sdpa=True,
    use_rope=True,
)
encoder_model.load_state_dict(encoder_state)
encoder_model = encoder_model.to(device).eval()


#%%
clip = prepare_clip(num_frames=NUM_FRAMES)
clip_batch = clip.unsqueeze(0)  # [1, C, T, H, W]
normalized_clip = (clip_batch - IMAGE_MEAN_255) / IMAGE_STD_255
normalized_clip = normalized_clip.to(device)


#%%
with torch.no_grad():
    encoder_tokens = encoder_model(normalized_clip)
    encoder_tokens = F.layer_norm(encoder_tokens, (encoder_tokens.size(-1),))

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)

    latents = torch.randn_like(normalized_clip)
    for t in scheduler.timesteps:
        timestep_tensor = torch.full((latents.size(0),), t, device=device, dtype=torch.long)
        latent_model_input = scheduler.scale_model_input(latents, t)
        noise_pred = decoder_model(
            noisy_clip=latent_model_input,
            timesteps=timestep_tensor,
            encoder_tokens=encoder_tokens,
        )
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    reconstructed = latents.cpu()

reconstructed_pixels = (reconstructed * IMAGE_STD_255) + IMAGE_MEAN_255
reconstructed_pixels = reconstructed_pixels.clamp(0.0, 255.0)

original_pixels = clip_batch.cpu()

mse = torch.mean((reconstructed_pixels - original_pixels) ** 2).item()
print(f"Reconstruction MSE (pixel space): {mse:.2f}")


#%%
frame_idx = 0
original_frame = (
    original_pixels[0, :, frame_idx].permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
)
reconstructed_frame = (
    reconstructed_pixels[0, :, frame_idx].permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original_frame)
axes[0].set_title("Original Frame")
axes[0].axis("off")

axes[1].imshow(reconstructed_frame)
axes[1].set_title("Diffusion Reconstructed Frame")
axes[1].axis("off")

plt.tight_layout()
plt.show()
