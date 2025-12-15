#%%
# /nvmessd/yinzi/vjepa2/checkpoints/decoder_train/latest.pt
# load
from pathlib import Path
import sys

import torch

# 添加路径
vjepa_root = Path("../../vjepa2")
if str(vjepa_root) not in sys.path:
    sys.path.insert(0, str(vjepa_root))
import src.models.vision_transformer as video_vit
from src.models.vjepa2_decoder import VJEPA2FrameDecoder

DEFAULT_CHECKPOINT = Path("/nvmessd/yinzi/vjepa2/checkpoints/decoder_train/latest.pt")
CHECKPOINT_PATH = Path("/nvmessd/yinzi/vjepa2/checkpoints/decoder_train_vitg/e75.pt")

if not CHECKPOINT_PATH.exists():
    print(f"{CHECKPOINT_PATH} not found. Falling back to {DEFAULT_CHECKPOINT}.")
    CHECKPOINT_PATH = DEFAULT_CHECKPOINT

print(f"Loading decoder checkpoint from {CHECKPOINT_PATH}")
checkpoint = torch.load(str(CHECKPOINT_PATH), map_location="cpu")
decoder_state_dict = checkpoint["decoder"]
encoder_state_dict = checkpoint["encoder"]

_ENCODER_VARIANTS = {
    384: ("vit_small", 6),
    768: ("vit_base", 12),
    1024: ("vit_large", 16),
    1280: ("vit_huge", 16),
    1408: ("vit_giant", 16),
    1664: ("vit_gigantic", 16),
}


def _infer_decoder_config(state_dict, default_embed_dim=1024, default_depth=24):
    decoder_embed_dim = state_dict.get("norm.weight")
    decoder_embed_dim = decoder_embed_dim.shape[0] if decoder_embed_dim is not None else default_embed_dim
    layer_ids = []
    prefix = "transformer.layers."
    for key in state_dict.keys():
        if key.startswith(prefix):
            parts = key.split(".")
            if len(parts) > 2:
                try:
                    layer_ids.append(int(parts[2]))
                except ValueError:
                    continue
    decoder_depth = max(layer_ids) + 1 if layer_ids else default_depth
    return decoder_embed_dim, decoder_depth


def _infer_encoder_config(state_dict):
    pos_embed = state_dict.get("pos_embed")
    if pos_embed is None:
        raise ValueError("Encoder state dict missing 'pos_embed'; cannot infer architecture.")
    embed_dim = pos_embed.shape[-1]
    model_name, num_heads = _ENCODER_VARIANTS.get(embed_dim, ("vit_large", 16))
    if model_name not in video_vit.__dict__:
        raise ValueError(f"Unsupported encoder embed dim {embed_dim}")
    return model_name, embed_dim, num_heads


decoder_embed_dim, decoder_depth = _infer_decoder_config(decoder_state_dict)
encoder_model_name, encoder_dim, decoder_num_heads = _infer_encoder_config(encoder_state_dict)
decoder_model = VJEPA2FrameDecoder(
    encoder_dim=encoder_dim,
    image_size=256,
    patch_size=16,
    tubelet_size=2,
    channels=3,
    decoder_embed_dim=decoder_embed_dim,
    depth=decoder_depth,
    num_heads=decoder_num_heads,
)

print(
    f"Decoder checkpoint implies encoder_dim={encoder_dim}, decoder_embed_dim={decoder_embed_dim}, depth={decoder_depth}"
)

load_msg = decoder_model.load_state_dict(decoder_state_dict, strict=False)
print(load_msg)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder_model = decoder_model.to(device).eval()

encoder_model = video_vit.__dict__[encoder_model_name](
    img_size=256,
    patch_size=16,
    num_frames=8,
    tubelet_size=2,
    uniform_power=True,
    use_sdpa=True,
    use_silu=False,
    wide_silu=True,
    use_activation_checkpointing=False,
    use_rope=True,
)
encoder_model.load_state_dict(encoder_state_dict)
encoder_model = encoder_model.to(device).eval()

print(f"Using encoder backbone '{encoder_model_name}'.")
print(f"Decoder model loaded to {device} and set to eval mode.")
print(f"Encoder model loaded to {device} and set to eval mode.")

IMAGE_MEAN_255 = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1) * 255.0
IMAGE_STD_255 = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1) * 255.0

# %%
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torchvision.io import read_video
from torchvision.transforms.functional import InterpolationMode, resize

try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except Exception as plt_import_error:
    print(f"Matplotlib unavailable ({plt_import_error}). Falling back to image file output.")
    _MATPLOTLIB_AVAILABLE = False

# DATASET_ROOT = Path("/nvmessd/yinzi/vjepa2/datasets/go_stanford_converted")
# CSV_PATH = DATASET_ROOT / "go_stanford_train_paths.csv"
DATASET_ROOT = Path("/nvmessd/yinzi/vjepa2/datasets/door_unity3d_collected_data")
CSV_PATH = DATASET_ROOT / "collected_data_paths.csv"

df = pd.read_csv(CSV_PATH, header=None)
sample_dir = Path(df.iloc[20, 0])

video_path = sample_dir / "recordings" / "MP4" / "nav_camera.mp4"
print(f"Using video: {video_path}")

frames, _, _ = read_video(str(video_path), pts_unit="sec")
if frames.size(0) == 0:
    raise RuntimeError(f"No frames found in {video_path}")

clip_len = min(8, frames.size(0))
if clip_len % 2 != 0:
    clip_len -= 1
if clip_len <= 0:
    raise RuntimeError("Not enough frames to create a clip with tubelet_size=2")
frames = frames[:clip_len].to(torch.float32)

# Convert to [T, C, H, W]
clip = frames.permute(0, 3, 1, 2)

# Resize each frame to 256 x 256 using bicubic interpolation
resized_frames = [
    resize(frame, [256, 256], interpolation=InterpolationMode.BICUBIC, antialias=True) for frame in clip
]
clip_resized = torch.stack(resized_frames, dim=0)

# Prepare tensors for model input
clip_model = clip_resized.permute(1, 0, 2, 3).contiguous()
clip_norm = (clip_model - IMAGE_MEAN_255) / IMAGE_STD_255
clip_batch = clip_norm.unsqueeze(0).to(device)

with torch.no_grad():
    tokens = encoder_model(clip_batch)
    tokens = F.layer_norm(tokens, (tokens.size(-1),))
    reconstructed_clip = decoder_model(tokens)

reconstructed_clip = reconstructed_clip.squeeze(0).cpu()
decoded_clip = (reconstructed_clip * IMAGE_STD_255) + IMAGE_MEAN_255
decoded_clip = decoded_clip.clamp(0.0, 255.0)

original_clip = clip_model.cpu()

frame_idx = 0
original_frame_uint8 = (
    original_clip[:, frame_idx].permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
)
decoded_frame_uint8 = (
    decoded_clip[:, frame_idx].permute(1, 2, 0).numpy().clip(0, 255).astype(np.uint8)
)
original_frame = original_frame_uint8.astype(np.float32) / 255.0
decoded_frame = decoded_frame_uint8.astype(np.float32) / 255.0

frame_mse = torch.mean((decoded_clip[:, frame_idx] - original_clip[:, frame_idx]) ** 2).item()
print(f"Frame {frame_idx} MSE (pixel space): {frame_mse:.2f}")

if _MATPLOTLIB_AVAILABLE:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_frame)
    axes[0].set_title("Original Frame")
    axes[0].axis("off")

    axes[1].imshow(decoded_frame)
    axes[1].set_title("Reconstructed Frame")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
else:
    from PIL import Image

    comparison_width = original_frame_uint8.shape[1] + decoded_frame_uint8.shape[1]
    comparison_height = original_frame_uint8.shape[0]
    comparison = Image.new("RGB", (comparison_width, comparison_height))
    comparison.paste(Image.fromarray(original_frame_uint8), (0, 0))
    comparison.paste(Image.fromarray(decoded_frame_uint8), (original_frame_uint8.shape[1], 0))
    output_path = Path("decoder_vis_demo_comparison.png").resolve()
    comparison.save(output_path)
    print(f"Saved visualization to {output_path}")

# %%
