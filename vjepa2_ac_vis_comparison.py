#%%
"""
V-JEPA2 AC Visualization Comparison Script

This script demonstrates:
1. Real frame encoded and decoded by V-JEPA2
2. Predicted frame from V-JEPA2 AC given first frame + action

Video: 12312.mp4 (frames from second 3-4)
Action: Forward
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.io import read_video
from torchvision.transforms.functional import InterpolationMode, resize

from app.vjepa_droid.transforms import make_transforms
from notebooks.utils.mpc_utils import compute_new_pose
from src.models.vjepa2_decoder import VJEPA2FrameDecoder
import src.models.vision_transformer as video_vit


def _strip_module_prefix(state_dict):
    return {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def infer_frame_decoder_config(state_dict, default_embed_dim=1024, default_depth=24):
    embed_param = state_dict.get("norm.weight")
    decoder_embed_dim = embed_param.shape[0] if embed_param is not None else default_embed_dim
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


def forward_target(encoder, clips, normalize_reps=True):
    """Encode video clips to representations."""
    B, C, T, H, W = clips.size()
    ctx = clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    reps = encoder(ctx)
    reps = reps.view(B, T, -1, reps.size(-1)).flatten(1, 2)
    if normalize_reps:
        reps = F.layer_norm(reps, (reps.size(-1),))
    return reps


# Configuration
VIDEO_PATH = Path("/nvmessd/yinzi/vjepa2/12312.mp4")
DECODER_CKPT = Path("/nvmessd/yinzi/vjepa2/checkpoints/decoder_train_vitg/e100.pt")
AC_CKPT_PATH = "/nvmessd/yinzi/vjepa2/checkpoints/dreamer_empty_finetune_4gpu/latest.pt"
CROP_SIZE = 256
NUM_FRAMES = 15  # Number of consecutive frames to visualize
OUTPUT_FIG_1 = Path("vjepa2_comparison_encode_decode.png")
OUTPUT_FIG_2 = Path("vjepa2_comparison_ac_predict.png")
OUTPUT_FIG_SEQUENCE = Path("vjepa2_encode_decode_sequence.png")

# Action parameters (Forward action)
ACTION_TRANSLATION = 0.2
ACTION_DELTA = np.array([ACTION_TRANSLATION, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
ACTION_LABEL = "Forward"

# Timing: extract frames from second 3-4 (assuming ~30fps)
START_SECOND = 3
END_SECOND = 4

#%%
# Load video and extract frames
print(f"Loading video: {VIDEO_PATH}")
frames_all, audio, info = read_video(str(VIDEO_PATH), pts_unit="sec")
fps = info.get("video_fps", 30.0)
print(f"Video info: {frames_all.shape[0]} frames, {fps} fps")

# Calculate frame indices for second 3-4
start_frame = int(START_SECOND * fps)
end_frame = int(END_SECOND * fps)
print(f"Extracting frames {start_frame} to {end_frame} (second {START_SECOND}-{END_SECOND})")

# Get first two frames for comparison
frame1_idx = start_frame
frame2_idx = start_frame + 1

# Get NUM_FRAMES consecutive frames for sequence visualization
last_frame_idx = start_frame + NUM_FRAMES - 1
if last_frame_idx >= frames_all.shape[0]:
    raise RuntimeError(f"Video too short. Need at least {last_frame_idx + 1} frames, got {frames_all.shape[0]}")

frame1 = frames_all[frame1_idx].numpy()  # H, W, C
frame2 = frames_all[frame2_idx].numpy()  # H, W, C

# Extract all NUM_FRAMES consecutive frames
frames_sequence = [frames_all[start_frame + i].numpy() for i in range(NUM_FRAMES)]

print(f"Frame 1 (idx {frame1_idx}): {frame1.shape}")
print(f"Frame 2 (idx {frame2_idx}): {frame2.shape}")
print(f"Extracted {NUM_FRAMES} consecutive frames for sequence visualization")

#%%
# Setup device and load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load V-JEPA2 AC model from torch hub
print("Loading V-JEPA2 AC model from torch hub...")
encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")

# Load fine-tuned checkpoint
print(f"Loading fine-tuned checkpoint from {AC_CKPT_PATH}...")
checkpoint = torch.load(AC_CKPT_PATH, map_location=device)
encoder.load_state_dict(_strip_module_prefix(checkpoint["encoder"]), strict=False)
predictor.load_state_dict(_strip_module_prefix(checkpoint["predictor"]), strict=False)
encoder = encoder.to(device).eval()
predictor = predictor.to(device).eval()

#%%
# Load decoder model
print(f"Loading decoder from {DECODER_CKPT}...")
decoder_device = torch.device("cpu") if device.type == "cuda" else device

decoder_checkpoint = torch.load(DECODER_CKPT, map_location=decoder_device)
decoder_state_dict = decoder_checkpoint.get("decoder", decoder_checkpoint)
decoder_embed_dim, decoder_depth = infer_frame_decoder_config(decoder_state_dict)

decoder_model = VJEPA2FrameDecoder(
    encoder_dim=encoder.embed_dim,
    image_size=CROP_SIZE,
    patch_size=encoder.patch_size,
    tubelet_size=getattr(encoder, "tubelet_size", 1),
    channels=3,
    decoder_embed_dim=decoder_embed_dim,
    depth=decoder_depth,
    num_heads=16,
).to(decoder_device)
decoder_model.load_state_dict(decoder_state_dict, strict=False)
decoder_model.eval()
print(f"Decoder loaded (embed_dim={decoder_embed_dim}, depth={decoder_depth})")

# Image normalization constants
IMAGE_MEAN_255 = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=decoder_device).view(1, 3, 1, 1, 1) * 255.0
IMAGE_STD_255 = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=decoder_device).view(1, 3, 1, 1, 1) * 255.0

#%%
# Prepare transforms
transform = make_transforms(
    random_horizontal_flip=False,
    random_resize_aspect_ratio=(1.0, 1.0),
    random_resize_scale=(1.0, 1.0),
    reprob=0.0,
    auto_augment=False,
    motion_shift=False,
    crop_size=CROP_SIZE,
)

tokens_per_frame = int((CROP_SIZE // encoder.patch_size) ** 2)

#%%
# Process frames
# Transform frame 1 and frame 2
clip1 = transform(np.expand_dims(frame1, axis=0)).unsqueeze(0).to(device)  # [1, C, 1, H, W]
clip2 = transform(np.expand_dims(frame2, axis=0)).unsqueeze(0).to(device)  # [1, C, 1, H, W]

print(f"Clip 1 shape: {clip1.shape}")
print(f"Clip 2 shape: {clip2.shape}")

#%%
# Encoding and decoding
with torch.no_grad():
    # Encode frame 1
    tokens1 = forward_target(encoder, clip1)
    frame1_tokens = tokens1[:, :tokens_per_frame]

    # Encode frame 2 (ground truth)
    tokens2 = forward_target(encoder, clip2)
    frame2_tokens = tokens2[:, :tokens_per_frame]

    # Decode real frame 2 encoding
    decoded_frame2_real = decoder_model(frame2_tokens.to(decoder_device))
    decoded_frame2_real = (decoded_frame2_real * IMAGE_STD_255) + IMAGE_MEAN_255
    decoded_frame2_real = decoded_frame2_real.clamp(0.0, 255.0)
    decoded_frame2_real_np = (
        decoded_frame2_real[:, :, 0]
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    # Predict frame 2 using V-JEPA2 AC (frame1 + action -> predicted frame2)
    action_vector = torch.tensor(ACTION_DELTA, dtype=torch.float32, device=device).view(1, 1, 7)
    initial_pose = torch.zeros((1, 1, 7), dtype=torch.float32, device=device)

    predicted_tokens = predictor(frame1_tokens, action_vector, initial_pose)[:, -tokens_per_frame:]

    # Decode predicted frame
    decoded_frame2_predicted = decoder_model(predicted_tokens.to(decoder_device))
    decoded_frame2_predicted = (decoded_frame2_predicted * IMAGE_STD_255) + IMAGE_MEAN_255
    decoded_frame2_predicted = decoded_frame2_predicted.clamp(0.0, 255.0)
    decoded_frame2_predicted_np = (
        decoded_frame2_predicted[:, :, 0]
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

print("Encoding and decoding completed!")

#%%
# Process and decode all NUM_FRAMES consecutive frames for sequence visualization
print(f"Processing {NUM_FRAMES} consecutive frames for sequence visualization...")
decoded_frames_sequence = []

with torch.no_grad():
    for i, frame in enumerate(frames_sequence):
        # Transform frame
        clip = transform(np.expand_dims(frame, axis=0)).unsqueeze(0).to(device)  # [1, C, 1, H, W]

        # Encode
        tokens = forward_target(encoder, clip)
        frame_tokens = tokens[:, :tokens_per_frame]

        # Decode
        decoded_frame = decoder_model(frame_tokens.to(decoder_device))
        decoded_frame = (decoded_frame * IMAGE_STD_255) + IMAGE_MEAN_255
        decoded_frame = decoded_frame.clamp(0.0, 255.0)
        decoded_frame_np = (
            decoded_frame[:, :, 0]
            .squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        decoded_frames_sequence.append(decoded_frame_np)

        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{NUM_FRAMES} frames")

print(f"Processed all {NUM_FRAMES} frames!")

#%%
# Prepare display frame 1 (resized to CROP_SIZE)
frame1_display = resize(
    torch.from_numpy(frame1).permute(2, 0, 1),
    [CROP_SIZE, CROP_SIZE],
    interpolation=InterpolationMode.BICUBIC,
    antialias=True
).permute(1, 2, 0).numpy().astype(np.uint8)

frame2_display = resize(
    torch.from_numpy(frame2).permute(2, 0, 1),
    [CROP_SIZE, CROP_SIZE],
    interpolation=InterpolationMode.BICUBIC,
    antialias=True
).permute(1, 2, 0).numpy().astype(np.uint8)

#%%
# Visualization 1: Real frame 2 encoded and decoded
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))

axes1[0].imshow(frame1_display)
axes1[0].set_title(f'Frame 1 (t={frame1_idx})\nAction: "{ACTION_LABEL}"', fontsize=12)
axes1[0].axis("off")

axes1[1].imshow(decoded_frame2_real_np)
axes1[1].set_title(f'Frame 2 (t={frame2_idx})\V-JEPA2', fontsize=12)
axes1[1].axis("off")

fig1.suptitle("V-JEPA2 Second Frame", fontsize=14)
plt.tight_layout()
fig1.savefig(OUTPUT_FIG_1, dpi=150, bbox_inches='tight')
print(f"Saved visualization 1 to {OUTPUT_FIG_1}")
plt.show()

#%%
# Visualization 2: V-JEPA2 AC prediction
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

axes2[0].imshow(frame1_display)
axes2[0].set_title(f'Frame 1 (t={frame1_idx})\nAction: "{ACTION_LABEL}"', fontsize=12)
axes2[0].axis("off")

axes2[1].imshow(decoded_frame2_predicted_np)
axes2[1].set_title(f'Predicted Frame 2\nV-JEPA2-AC (Frame1 + Action) Decoded', fontsize=12)
axes2[1].axis("off")

fig2.suptitle("V-JEPA2-AC Prediction: Predicted Next Frame", fontsize=14)
plt.tight_layout()
fig2.savefig(OUTPUT_FIG_2, dpi=150, bbox_inches='tight')
print(f"Saved visualization 2 to {OUTPUT_FIG_2}")
plt.show()

#%%
# Combined visualization (both comparisons in one figure)
fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))

# Row 1: Encode-Decode comparison
axes3[0, 0].imshow(frame1_display)
axes3[0, 0].set_title(f'Frame 1 (t={frame1_idx})\nAction: "{ACTION_LABEL}"', fontsize=11)
axes3[0, 0].axis("off")

axes3[0, 1].imshow(decoded_frame2_real_np)
axes3[0, 1].set_title(f'Frame 2 (t={frame2_idx})\nV-JEPA2', fontsize=11)
axes3[0, 1].axis("off")

# Row 2: AC Prediction comparison
axes3[1, 0].imshow(frame1_display)
axes3[1, 0].set_title(f'Frame 1 (t={frame1_idx})\nAction: "{ACTION_LABEL}"', fontsize=11)
axes3[1, 0].axis("off")

axes3[1, 1].imshow(decoded_frame2_predicted_np)
axes3[1, 1].set_title('V-JEPA2-AC Predicted Frame\n(Frame1 + Action) Decoded', fontsize=11)
axes3[1, 1].axis("off")

fig3.suptitle("V-JEPA2-AC (Video: 12312.mp4, Second 3-4)", fontsize=14)
plt.tight_layout()
fig3.savefig("vjepa2_comparison_combined.png", dpi=150, bbox_inches='tight')
print("Saved combined visualization to vjepa2_comparison_combined.png")
plt.show()

#%%
# Visualization 4: Sequence of NUM_FRAMES consecutive frames (encode + decode)
# First frame is displayed directly, subsequent frames are encoder+decoder outputs
# Layout: 3 rows x 5 columns for 15 frames
n_cols = 5
n_rows = (NUM_FRAMES + n_cols - 1) // n_cols  # Ceiling division

fig4, axes4 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
axes4 = axes4.flatten()

# First frame: display directly (resized original)
frame1_seq_display = resize(
    torch.from_numpy(frames_sequence[0]).permute(2, 0, 1),
    [CROP_SIZE, CROP_SIZE],
    interpolation=InterpolationMode.BICUBIC,
    antialias=True
).permute(1, 2, 0).numpy().astype(np.uint8)

axes4[0].imshow(frame1_seq_display)
axes4[0].set_title('Frame 1', fontsize=11)
axes4[0].axis("off")

# Subsequent frames: encoder + decoder outputs with t+1, t+2, ... labels
for i in range(1, NUM_FRAMES):
    axes4[i].imshow(decoded_frames_sequence[i])
    axes4[i].set_title(f't+{i}', fontsize=11)
    axes4[i].axis("off")

# Hide any extra subplots if NUM_FRAMES doesn't fill the grid
for i in range(NUM_FRAMES, len(axes4)):
    axes4[i].axis("off")

fig4.suptitle("V-JEPA2 Encode-Decode Sequence (15 Consecutive Frames)", fontsize=14)
plt.tight_layout()
fig4.savefig(OUTPUT_FIG_SEQUENCE, dpi=150, bbox_inches='tight')
print(f"Saved sequence visualization to {OUTPUT_FIG_SEQUENCE}")
plt.show()

print("\nDone!")
# %%
