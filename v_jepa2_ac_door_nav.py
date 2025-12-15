#%%
import sys

sys.path.insert(0, "..")

#%%
from pathlib import Path
import json

import h5py
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.nn import functional as F

from app.vjepa_droid.transforms import make_transforms
from notebooks.utils.mpc_utils import compute_new_pose
from src.models.vjepa2_decoder import VJEPA2FrameDecoder


def _strip_module_prefix(state_dict):
    return {
        key.replace("module.", "", 1) if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


DATASET_ROOT = Path("/nvmessd/yinzi/vjepa2/datasets/door_unity3d_collected_data")
VIDEO_PREVIEW_START = 30


def forward_target(clips, normalize_reps=True):
    B, C, T, H, W = clips.size()
    ctx = clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
    reps = encoder(ctx)
    reps = reps.view(B, T, -1, reps.size(-1)).flatten(1, 2)
    if normalize_reps:
        reps = F.layer_norm(reps, (reps.size(-1),))
    return reps


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


def resolve_video_path(ep_dir: Path):
    metadata = json.loads(ep_dir.joinpath("metadata.json").read_text())
    video_rel = (
        metadata.get("video_path")
        or metadata.get("left_mp4_path")
        or metadata.get("right_mp4_path")
        or "recordings/MP4/nav_camera.mp4"
    )
    return ep_dir / video_rel


def load_sample_frame(dataset_root: Path, frame_index: int = 0):
    episode_dirs = sorted(dataset_root.glob("episode_*"))
    if not episode_dirs:
        raise RuntimeError(f"No door episodes found in {dataset_root}")
    episode_dir = episode_dirs[0]
    video_path = resolve_video_path(episode_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video file: {video_path}")

    reader = imageio.get_reader(video_path)
    try:
        total_frames = None
        try:
            total_frames = reader.count_frames()
        except Exception:
            total_frames = None
        if total_frames is not None and frame_index >= total_frames:
            raise ValueError(f"Requested frame {frame_index} but video only has {total_frames} frames.")
        frame = reader.get_data(frame_index)
    finally:
        reader.close()

    return frame, episode_dir.name, video_path


#%%
# Preview Door Unity3D dataset and video frames
try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover
    pd = None

try:
    import ipywidgets as widgets  # type: ignore
except ImportError:  # pragma: no cover
    widgets = None

try:
    from IPython.display import display as ipy_display, clear_output as ipy_clear_output
except ImportError:  # pragma: no cover
    ipy_display = None
    ipy_clear_output = None


episode_dirs = sorted(DATASET_ROOT.glob("episode_*"))
if not episode_dirs:
    raise RuntimeError(f"No door episodes found in {DATASET_ROOT}")

print(f"Dataset root: {DATASET_ROOT}")
print(f"Total episodes: {len(episode_dirs)}")


def _load_episode_contents(ep_dir: Path):
    metadata = json.loads(ep_dir.joinpath("metadata.json").read_text())
    with h5py.File(ep_dir / "trajectory.h5", "r") as f:
        cartesian = f["observation/robot_state/cartesian_position"][:]
        gripper = f["observation/robot_state/gripper_position"][:]
    return metadata, cartesian, gripper


max_episodes_to_show = 3
episode_summaries = []
timeseries = []
for ep_dir in episode_dirs[:max_episodes_to_show]:
    metadata, cartesian, gripper = _load_episode_contents(ep_dir)
    episode_summaries.append(
        {
            "episode": ep_dir.name,
            "length": cartesian.shape[0],
            "start_pose": cartesian[0].tolist() if len(cartesian) else None,
            "end_pose": cartesian[-1].tolist() if len(cartesian) else None,
            "avg_grip": float(np.mean(gripper)) if len(gripper) else None,
        }
    )
    timeseries.append((ep_dir.name, cartesian, gripper))

if pd is not None and ipy_display is not None:
    ipy_display(pd.DataFrame(episode_summaries))
else:
    from pprint import pprint

    pprint(episode_summaries)

if not timeseries:
    raise RuntimeError("No episode summaries available to visualize")

num_rows = len(timeseries)
fig, axes = plt.subplots(num_rows, 2, figsize=(14, 4 * num_rows), sharex="col")
if num_rows == 1:
    axes = axes[np.newaxis, :]

for idx, (name, cartesian, gripper) in enumerate(timeseries):
    steps = np.arange(cartesian.shape[0])
    axes[idx, 0].plot(steps, cartesian[:, 0], label="x", color="tab:blue")
    axes[idx, 0].plot(steps, cartesian[:, 1], label="y", color="tab:orange")
    axes[idx, 0].plot(steps, cartesian[:, 2], label="z", color="tab:green")
    axes[idx, 0].set_ylabel("Position (m)")
    axes[idx, 0].set_xlabel("Timestep")
    axes[idx, 0].set_title(f"{name}: Cartesian positions")
    axes[idx, 0].grid(True, alpha=0.25)
    axes[idx, 0].legend(loc="upper right")

    axes[idx, 1].plot(steps, gripper, color="tab:red")
    axes[idx, 1].set_ylabel("Gripper")
    axes[idx, 1].set_xlabel("Timestep")
    axes[idx, 1].set_title(f"{name}: Gripper position")
    axes[idx, 1].grid(True, alpha=0.25)

fig.suptitle("Door Unity3D dataset overview")
fig.tight_layout()
plt.show()

sample_episode_dir = episode_dirs[0]
sample_video_path = resolve_video_path(sample_episode_dir)
print("\nSample episode video preview:")
print(f"Episode: {sample_episode_dir.name}")
print(f"Video path: {sample_video_path}")
if not sample_video_path.exists():
    print(f"Video file not found: {sample_video_path}")
else:
    try:
        base_reader = imageio.get_reader(sample_video_path)
    except Exception as err:  # pragma: no cover
        print(f"Failed to open video: {err}")
    else:
        try:
            total_frames = base_reader.count_frames()
        except Exception:
            total_frames = None
        base_reader.close()

        frames_to_show = 6

        def show_sample_frames(start_frame=VIDEO_PREVIEW_START):
            with imageio.get_reader(sample_video_path) as reader:
                start_idx = max(0, int(start_frame))
                if total_frames is not None and total_frames > 0:
                    start_idx = min(start_idx, max(total_frames - frames_to_show, 0))
                sampled_frames = []
                for frame_idx in range(start_idx, start_idx + frames_to_show):
                    try:
                        frame = reader.get_data(frame_idx)
                    except Exception:
                        break
                    sampled_frames.append((frame_idx, frame))

            if not sampled_frames:
                print(f"No frames decoded starting from t={start_idx}.")
                return

            fig_frames, axes_frames = plt.subplots(
                1, len(sampled_frames), figsize=(4 * len(sampled_frames), 4)
            )
            if len(sampled_frames) == 1:
                axes_frames = [axes_frames]
            for ax, (frame_idx, frame) in zip(axes_frames, sampled_frames):
                ax.imshow(frame)
                ax.axis("off")
                ax.set_title(f"frame {frame_idx}")
            first_frame_label = sampled_frames[0][0]
            fig_frames.suptitle(
                f"{sample_episode_dir.name} video frames (t≥{first_frame_label})"
            )
            fig_frames.tight_layout()
            plt.show()

        if total_frames is None or total_frames <= 0:
            print("Video length unavailable; showing frames starting at t=30 by default.")
            show_sample_frames(start_frame=VIDEO_PREVIEW_START)
        elif widgets is None or ipy_display is None or ipy_clear_output is None:
            if widgets is None:
                print("ipywidgets not installed; showing static preview from t=30.")
            show_sample_frames(start_frame=VIDEO_PREVIEW_START)
        else:
            slider = widgets.IntSlider(
                min=0,
                max=max(total_frames - frames_to_show, 0),
                value=min(VIDEO_PREVIEW_START, max(total_frames - frames_to_show, 0)),
                description="start t",
                continuous_update=False,
            )
            output = widgets.Output()
            ipy_display(widgets.VBox([slider]), output)

            def on_slider_change(change):
                if change.get("name") == "value":
                    with output:
                        ipy_clear_output(wait=True)
                        show_sample_frames(start_frame=change["new"])

            slider.observe(on_slider_change, names="value")

            with output:
                show_sample_frames(start_frame=slider.value)


#%%
# Initialize VJEPA 2-AC model
encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")


#%%
# Load Door Unity3D fine-tuned checkpoint
checkpoint_path = "/nvmessd/yinzi/vjepa2/checkpoints/door_finetune_4gpu/latest.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(checkpoint_path, map_location=device)

if "encoder" not in checkpoint or "predictor" not in checkpoint:
    raise KeyError(
        f"Checkpoint {checkpoint_path} is missing encoder/predictor weights: {checkpoint.keys()}"
    )

encoder.load_state_dict(_strip_module_prefix(checkpoint["encoder"]), strict=False)
predictor.load_state_dict(_strip_module_prefix(checkpoint["predictor"]), strict=False)

encoder = encoder.to(device).eval()
predictor = predictor.to(device).eval()

print(f"Loaded Door Unity3D checkpoint from {checkpoint_path} on {device}")


#%%
# Prepare transforms and constants for the synthetic forward rollout
DECODER_CKPT = Path("/nvmessd/yinzi/vjepa2/checkpoints/decoder_train_vitg/e100.pt")
CROP_SIZE = 256
FORWARD_TRANSLATION = 0.2
NUM_FORWARD_STEPS = 5
FRAME_INDEX = VIDEO_PREVIEW_START
OUTPUT_FIG = Path("door_nav_rollout.png")
decoder_device = torch.device("cpu") if device.type == "cuda" else device


tokens_per_frame = int((CROP_SIZE // encoder.patch_size) ** 2)
transform = make_transforms(
    random_horizontal_flip=False,
    random_resize_aspect_ratio=(1.0, 1.0),
    random_resize_scale=(1.0, 1.0),
    reprob=0.0,
    auto_augment=False,
    motion_shift=False,
    crop_size=CROP_SIZE,
)


#%%
# Load the frame decoder to visualize predictor latents
if not DECODER_CKPT.exists():
    raise FileNotFoundError(f"Missing decoder checkpoint at {DECODER_CKPT}")


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
decoder_load_msg = decoder_model.load_state_dict(decoder_state_dict, strict=False)
decoder_model.eval()
print(
    f"Loaded decoder from {DECODER_CKPT} on {decoder_device} (embed_dim={decoder_embed_dim}, depth={decoder_depth}): "
    f"{decoder_load_msg}"
)

IMAGE_MEAN_255 = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=decoder_device).view(1, 3, 1, 1, 1) * 255.0
IMAGE_STD_255 = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=decoder_device).view(1, 3, 1, 1, 1) * 255.0


#%%
# Sample a frame and run the forward-action rollout
raw_frame, episode_name, video_source = load_sample_frame(DATASET_ROOT, frame_index=FRAME_INDEX)
print(f"Using episode {episode_name} frame {FRAME_INDEX} from {video_source}")

clip = np.expand_dims(raw_frame.astype(np.uint8), axis=0)  # [T=1, H, W, C]
clips = transform(clip).unsqueeze(0).to(device)  # [1, C, T, H, W]

initial_pose = torch.zeros((1, 1, 7), dtype=torch.float32, device=device)
forward_action = torch.zeros((1, 1, 7), dtype=torch.float32, device=device)
forward_action[..., 0] = FORWARD_TRANSLATION  # positive X -> move forward

print(f"Applying {NUM_FORWARD_STEPS} synthetic forward actions with delta {FORWARD_TRANSLATION:.2f} along X.")

with torch.no_grad():
    latents = forward_target(clips)
    current_tokens = latents[:, :tokens_per_frame]
    current_pose = initial_pose
    rollout_tokens = []

    for step in range(NUM_FORWARD_STEPS):
        predicted_tokens = predictor(current_tokens, forward_action, current_pose)[:, -tokens_per_frame:]
        rollout_tokens.append(predicted_tokens)
        current_tokens = predicted_tokens
        current_pose = compute_new_pose(current_pose, forward_action)
        print(f"Step {step + 1}: pose -> {current_pose.squeeze().tolist()}")

    final_tokens = rollout_tokens[-1].to(decoder_device)
    reconstructed_clip = decoder_model(final_tokens)


decoded_pixels = (reconstructed_clip * IMAGE_STD_255) + IMAGE_MEAN_255
decoded_pixels = decoded_pixels.clamp(0.0, 255.0)
decoded_frame = (
    decoded_pixels[:, :, 0]
    .squeeze(0)
    .permute(1, 2, 0)
    .cpu()
    .numpy()
    .astype(np.uint8)
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(raw_frame)
axes[0].set_title(f"Input Frame (t={FRAME_INDEX})")
axes[0].axis("off")

axes[1].imshow(decoded_frame)
axes[1].set_title(f"Decoder Output after {NUM_FORWARD_STEPS}×Forward")
axes[1].axis("off")

plt.suptitle("Door Unity3D: synthetic forward rollout vs. decoder reconstruction")
plt.tight_layout()
fig.savefig(OUTPUT_FIG, dpi=150)
plt.show()
