#!/usr/bin/env python3
"""
Convert RLDS format DROID dataset to V-JEPA2 compatible format
"""

import os
import json
import h5py
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
from tqdm import tqdm
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_video(output_path, frames, fps=15):
    """Save frames as MP4 video using OpenCV"""
    if len(frames) == 0:
        logger.warning(f"No frames to save for {output_path}")
        return
    
    output_path = str(output_path)
    height, width = frames[0].shape[:2]
    
    # Use H264 codec for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        logger.error(f"Failed to open video writer for {output_path}")
        return
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    logger.info(f"Saved video: {output_path} ({len(frames)} frames)")


def create_metadata(episode_dir, camera_names):
    """Create metadata.json file"""
    metadata = {}
    
    # Map camera names to expected format
    if "exterior_image_1_left" in camera_names:
        metadata["left_mp4_path"] = "recordings/MP4/exterior_image_1_left.mp4"
        metadata["right_mp4_path"] = "recordings/MP4/exterior_image_1_left.mp4"  # Reuse left
    
    if "wrist_image_left" in camera_names:
        metadata["wrist_mp4_path"] = "recordings/MP4/wrist_image_left.mp4"
    
    metadata_path = episode_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Created metadata: {metadata_path}")


def create_trajectory_h5(episode_dir, cartesian_positions, gripper_positions, camera_names):
    """Create trajectory.h5 file"""
    h5_path = episode_dir / "trajectory.h5"
    
    with h5py.File(h5_path, "w") as f:
        obs = f.create_group("observation")
        robot_state = obs.create_group("robot_state")
        
        # Save trajectory data
        cartesian_data = np.array(cartesian_positions, dtype=np.float64)  # [T, 6]
        gripper_data = np.array(gripper_positions, dtype=np.float64).squeeze()  # [T]
        
        if cartesian_data.ndim == 1:
            cartesian_data = cartesian_data.reshape(-1, 6)
        
        robot_state.create_dataset("cartesian_position", data=cartesian_data)
        robot_state.create_dataset("gripper_position", data=gripper_data)
        
        # Create camera extrinsics (using zero matrices)
        camera_extrinsics = obs.create_group("camera_extrinsics")
        num_frames = len(cartesian_positions)
        zero_extrinsics = np.zeros((num_frames, 6), dtype=np.float64)
        
        # Create extrinsics for each camera
        for camera_name in camera_names:
            # V-JEPA2 expects camera_name_left format
            extrinsics_name = f"{camera_name.replace('.mp4', '')}_left"
            camera_extrinsics.create_dataset(extrinsics_name, data=zero_extrinsics)
    
    logger.info(f"Created trajectory: {h5_path}")


def convert_episode(episode, episode_idx, output_dir, fps=15):
    """Convert a single episode from RLDS to V-JEPA2 format"""
    episode_dir = Path(output_dir) / f"episode_{episode_idx:06d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store different camera views
    camera_frames = {
        "exterior_image_1_left": [],
        "wrist_image_left": []
    }
    
    # Lists to store trajectory data
    cartesian_positions = []
    gripper_positions = []
    
    # Collect all steps data
    step_count = 0
    for step in episode["steps"]:
        # Collect images
        for camera_name in camera_frames.keys():
            if camera_name in step["observation"]:
                frame = step["observation"][camera_name].numpy()
                camera_frames[camera_name].append(frame)
        
        # Collect trajectory data
        cart_pos = step["observation"]["cartesian_position"].numpy()
        grip_pos = step["observation"]["gripper_position"].numpy()
        
        cartesian_positions.append(cart_pos)
        gripper_positions.append(grip_pos)
        step_count += 1
    
    logger.info(f"Episode {episode_idx}: {step_count} steps")
    
    # Create video directory
    video_dir = episode_dir / "recordings" / "MP4"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Save videos for each camera
    saved_cameras = []
    for camera_name, frames in camera_frames.items():
        if len(frames) > 0:
            video_path = video_dir / f"{camera_name}.mp4"
            save_video(video_path, frames, fps)
            saved_cameras.append(camera_name)
    
    # Create metadata.json
    create_metadata(episode_dir, saved_cameras)
    
    # Create trajectory.h5
    create_trajectory_h5(episode_dir, cartesian_positions, gripper_positions, saved_cameras)
    
    return str(episode_dir)


def create_csv_file(output_dir, episode_paths):
    """Create CSV index file"""
    csv_path = Path(output_dir) / "droid_train_paths.csv"
    
    with open(csv_path, "w") as f:
        for path in episode_paths:
            f.write(f"{path}\n")
    
    logger.info(f"Created CSV file: {csv_path} with {len(episode_paths)} episodes")


def main():
    parser = argparse.ArgumentParser(description="Convert RLDS DROID to V-JEPA2 format")
    parser.add_argument("--input_dir", type=str, default="/nvmessd/yinzi/navigation_datasets/DROID_raw",
                        help="Path to RLDS dataset root (parent of droid_100)")
    parser.add_argument("--output_dir", type=str, default="/nvmessd/yinzi/vjepa2/droid_converted",
                        help="Output directory for converted data")
    parser.add_argument("--max_episodes", type=int, default=5,
                        help="Maximum number of episodes to convert (for testing)")
    parser.add_argument("--fps", type=int, default=15,
                        help="FPS for output videos")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading DROID dataset from: {args.input_dir}")
    
    # Load the dataset using TFDS
    try:
        # Set the correct data directory
        os.environ['TFDS_DATA_DIR'] = args.input_dir
        
        # Load the dataset
        ds = tfds.load("droid_100", 
                      data_dir=args.input_dir,
                      split="train",
                      download=False,
                      as_supervised=False)
        logger.info("Successfully loaded dataset")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        
        # Try another approach: use builder_from_directory
        try:
            logger.info("Trying builder_from_directory approach...")
            builder_path = Path(args.input_dir) / "droid_100" / "1.0.0"
            builder = tfds.builder_from_directory(str(builder_path))
            ds = builder.as_dataset(split="train")
            logger.info("Successfully loaded dataset using builder_from_directory")
        except Exception as e2:
            logger.error(f"Failed with builder_from_directory: {e2}")
            raise
    
    episode_paths = []
    
    # Convert episodes
    logger.info(f"Converting up to {args.max_episodes} episodes...")
    
    with tqdm(total=args.max_episodes) as pbar:
        for episode_idx, episode in enumerate(ds.take(args.max_episodes)):
            try:
                episode_path = convert_episode(episode, episode_idx, output_dir, args.fps)
                episode_paths.append(episode_path)
                pbar.update(1)
            except Exception as e:
                logger.error(f"Failed to convert episode {episode_idx}: {e}")
                continue
    
    # Create CSV file
    if episode_paths:
        create_csv_file(output_dir, episode_paths)
        logger.info(f"Successfully converted {len(episode_paths)} episodes")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"CSV file: {output_dir}/droid_train_paths.csv")
    else:
        logger.error("No episodes were successfully converted")


if __name__ == "__main__":
    main()