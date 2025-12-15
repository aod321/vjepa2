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


DATASET_ROOT = Path("/nvmessd/yinzi/vjepa2/datasets/dreamer_empty_unity")
VIDEO_PREVIEW_START = 30
ROLLOUT_STEPS = 30
EPISODE_INDEX = 0  # index into the sorted episode list
USE_RECORDED_ACTIONS = False
SYNTHETIC_ACTION_ID = 5  # forward action id

ACTION_TRANSLATION = 0.2
ACTION_ROTATION = np.deg2rad(15.0)
ACTION_LABELS = {
    0: "stop",
    1: "forward",
    2: "backward",
    3: "strafe_left",
    4: "strafe_right",
    5: "turn_left",
    6: "turn_right",
}
ACTION_ID_TO_DELTA = {
    0: np.zeros(7, dtype=np.float32),
    1: np.array([ACTION_TRANSLATION, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    2: np.array([-ACTION_TRANSLATION, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    3: np.array([0.0, -ACTION_TRANSLATION, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    4: np.array([0.0, ACTION_TRANSLATION, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    5: np.array([0.0, 0.0, 0.0, 0.0, 0.0, ACTION_ROTATION, 0.0], dtype=np.float32),
    6: np.array([0.0, 0.0, 0.0, 0.0, 0.0, -ACTION_ROTATION, 0.0], dtype=np.float32),
}


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


def load_sample_frame(dataset_root: Path, frame_index: int = 0):
    episode_dirs = sorted(dataset_root.glob("episode_*"))
    if not episode_dirs:
        raise RuntimeError(f"No Dreamer Empty episodes found in {dataset_root}")
    episode_dir = episode_dirs[0]
    video_path = episode_dir / "recordings/MP4/episode_camera.mp4"
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


def action_id_to_pose_delta(action_id: int, device: torch.device):
    base = ACTION_ID_TO_DELTA.get(int(action_id), ACTION_ID_TO_DELTA[0])
    tensor = torch.tensor(base, dtype=torch.float32, device=device).view(1, 1, 7)
    return tensor


def load_episode_sequence(
    dataset_root: Path,
    episode_idx: int,
    start_frame: int,
    rollout_steps: int,
):
    episode_dirs = sorted(dataset_root.glob("episode_*"))
    if not episode_dirs:
        raise RuntimeError(f"No Dreamer Empty episodes found in {dataset_root}")

    episode_dir = episode_dirs[episode_idx % len(episode_dirs)]
    metadata, actions, _, _ = _load_episode_contents(episode_dir)
    video_rel = metadata.get("video_path", "recordings/MP4/episode_camera.mp4")
    video_path = episode_dir / video_rel
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video file: {video_path}")

    total_actions = len(actions)
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if total_actions < rollout_steps:
        raise ValueError(f"Episode {episode_dir.name} too short for {rollout_steps} steps: {total_actions} actions")

    start_idx = int(np.clip(start_frame, 0, total_actions - rollout_steps))

    with imageio.get_reader(video_path) as reader:
        try:
            total_frames = reader.count_frames()
        except Exception:
            total_frames = None

        if total_frames is not None:
            max_video_start = max(total_frames - (rollout_steps + 1), 0)
            start_idx = min(start_idx, max_video_start)

        frames = []
        for frame_idx in range(start_idx, start_idx + rollout_steps + 1):
            frames.append(reader.get_data(frame_idx))

    frames_np = np.stack(frames, axis=0)
    seq_actions = actions[start_idx : start_idx + rollout_steps]
    return frames_np, seq_actions, metadata, episode_dir.name, video_path, start_idx


#%%
# Preview Dreamer Empty Unity dataset and video frames
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
    raise RuntimeError(f"No Dreamer Empty episodes found in {DATASET_ROOT}")

print(f"Dataset root: {DATASET_ROOT}")
print(f"Total episodes: {len(episode_dirs)}")


def _load_episode_contents(ep_dir: Path):
    metadata = json.loads(ep_dir.joinpath("metadata.json").read_text())
    with h5py.File(ep_dir / "trajectory.h5", "r") as f:
        actions = f["episode"]["actions"][:]
        rewards = f["episode"]["rewards"][:]
        door_counts = f["episode"]["door_counts"][:]
    return metadata, actions, rewards, door_counts


max_episodes_to_show = 3
episode_summaries = []
timeseries = []
for ep_dir in episode_dirs[:max_episodes_to_show]:
    metadata, actions, rewards, door_counts = _load_episode_contents(ep_dir)
    episode_summaries.append(
        {
            "episode": metadata.get("episode_name", ep_dir.name),
            "length": int(metadata.get("episode_length", len(actions))),
            "reward": float(metadata.get("episode_reward", 0.0)),
            "door_touches": int(metadata.get("total_door_touches", 0)),
            "first_actions": actions[:5].tolist(),
        }
    )
    timeseries.append((metadata.get("episode_name", ep_dir.name), actions, rewards, door_counts))

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

for idx, (name, actions, rewards, door_counts) in enumerate(timeseries):
    steps = np.arange(actions.shape[0])
    axes[idx, 0].step(steps, actions, where="mid", label="action id")
    axes[idx, 0].plot(steps, rewards, color="tab:orange", alpha=0.6, label="reward")
    axes[idx, 0].set_ylabel("Value")
    axes[idx, 0].set_xlabel("Timestep")
    axes[idx, 0].set_title(f"{name}: actions vs. rewards")
    axes[idx, 0].grid(True, alpha=0.25)
    axes[idx, 0].legend(loc="upper right")

    if door_counts.ndim == 2:
        for door_idx in range(door_counts.shape[1]):
            axes[idx, 1].plot(
                steps,
                door_counts[:, door_idx],
                label=f"door {door_idx}",
                alpha=0.8,
            )
    else:
        axes[idx, 1].plot(steps, door_counts, label="door counts", alpha=0.8)
    axes[idx, 1].set_ylabel("Count")
    axes[idx, 1].set_xlabel("Timestep")
    axes[idx, 1].set_title(f"{name}: door counts")
    axes[idx, 1].grid(True, alpha=0.25)
    axes[idx, 1].legend(loc="upper left", ncol=2, fontsize=8)

fig.suptitle("Dreamer Empty Unity dataset overview")
fig.tight_layout()
plt.show()

sample_episode_dir = episode_dirs[0]
sample_metadata_path = sample_episode_dir / "metadata.json"
sample_metadata = json.loads(sample_metadata_path.read_text())
sample_video_rel = sample_metadata.get("video_path", "recordings/MP4/episode_camera.mp4")
sample_video_path = sample_episode_dir / sample_video_rel
print("\nSample episode video preview:")
print(f"Episode: {sample_metadata.get('episode_name', sample_episode_dir.name)}")
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
                f"{sample_metadata.get('episode_name', sample_episode_dir.name)} video frames (t≥{first_frame_label})"
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
# Load Dreamer Empty Unity fine-tuned checkpoint
checkpoint_path = "/nvmessd/yinzi/vjepa2/checkpoints/dreamer_empty_finetune_4gpu/latest.pt"
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

print(f"Loaded Dreamer Empty Unity checkpoint from {checkpoint_path} on {device}")


#%%
# Prepare transforms and constants for the interactive rollout
DECODER_CKPT = Path("/nvmessd/yinzi/vjepa2/checkpoints/decoder_train_vitg/e100.pt")
CROP_SIZE = 256
FRAME_INDEX = VIDEO_PREVIEW_START
OUTPUT_FIG = Path("dreamer_empty_nav_rollout_latest_ckpt.png")
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
# Sample a sequence and replay actions
sequence_frames, recorded_actions, _, episode_name, video_source, start_idx = load_episode_sequence(
    DATASET_ROOT,
    episode_idx=EPISODE_INDEX,
    start_frame=FRAME_INDEX,
    rollout_steps=ROLLOUT_STEPS,
)

if USE_RECORDED_ACTIONS:
    sequence_actions = recorded_actions
    action_source = "recorded"
else:
    sequence_actions = np.full((ROLLOUT_STEPS,), SYNTHETIC_ACTION_ID, dtype=np.int64)
    action_source = f"synthetic id={SYNTHETIC_ACTION_ID}"

print(
    f"Episode {episode_name} | video={video_source}\n"
    f"Start frame: {start_idx} (showing {ROLLOUT_STEPS} steps)\n"
    f"Action source: {action_source}\n"
    f"Actions: {sequence_actions.tolist()}"
)

context_clip = transform(sequence_frames[:1]).unsqueeze(0).to(device)
initial_pose = torch.zeros((1, 1, 7), dtype=torch.float32, device=device)

with torch.no_grad():
    latents = forward_target(context_clip)
    current_tokens = latents[:, :tokens_per_frame]
    current_pose = initial_pose

    for step_idx, action_id in enumerate(sequence_actions):
        action_vector = action_id_to_pose_delta(int(action_id), device)
        current_tokens = predictor(current_tokens, action_vector, current_pose)[:, -tokens_per_frame:]
        current_pose = compute_new_pose(current_pose, action_vector)

        action_name = ACTION_LABELS.get(int(action_id), "unknown")
        pose_list = [round(x, 3) for x in current_pose.squeeze().tolist()]
        print(f"Step {step_idx + 1}: action={action_name} (id={int(action_id)}), pose={pose_list}")

    final_clip = decoder_model(current_tokens.to(decoder_device))
    decoded_pixels = (final_clip * IMAGE_STD_255) + IMAGE_MEAN_255
    decoded_pixels = decoded_pixels.clamp(0.0, 255.0)
    decoded_final_frame = (
        decoded_pixels[:, :, 0]
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

title_desc = "recorded" if USE_RECORDED_ACTIONS else "synthetic"
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(sequence_frames[0])
axes[0].set_title("Reference frame t=0")
axes[0].axis("off")

axes[1].imshow(decoded_final_frame)
axes[1].set_title(f"AC prediction t+{ROLLOUT_STEPS}")
axes[1].axis("off")

plt.suptitle(
    f"Dreamer Empty: {episode_name} {title_desc} rollout (final step view)", fontsize=14
)
plt.tight_layout()
fig.savefig(OUTPUT_FIG, dpi=150)
plt.show()

# %%
