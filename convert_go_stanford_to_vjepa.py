#!/usr/bin/env python3
"""
Convert go_stanford navigation dataset to V-JEPA2 DROID format
"""

import os
import json
import pickle
import h5py
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import logging
from multiprocessing import Pool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def compute_angle_diff(angle1, angle2):
    """Compute shortest angular distance from angle1 to angle2"""
    diff = angle2 - angle1
    return normalize_angle(diff)


def create_video_from_images(image_dir, output_path, fps=10):
    """Create MP4 video from image sequence"""
    # Get all jpg files and sort them numerically
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    image_files.sort(key=lambda x: int(x.replace('.jpg', '')))
    
    if len(image_files) == 0:
        logger.warning(f"No images found in {image_dir}")
        return False
    
    # Read first image to get dimensions
    first_img_path = os.path.join(image_dir, image_files[0])
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        logger.error(f"Failed to read {first_img_path}")
        return False
    
    height, width = first_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        logger.error(f"Failed to open video writer for {output_path}")
        return False
    
    # Write all frames
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        frame = cv2.imread(img_path)
        if frame is not None:
            out.write(frame)
        else:
            logger.warning(f"Failed to read {img_path}, skipping frame")
    
    out.release()
    return True


def create_metadata(episode_dir):
    """Create metadata.json file"""
    metadata = {
        "left_mp4_path": "recordings/MP4/nav_camera.mp4",
        "right_mp4_path": "recordings/MP4/nav_camera.mp4",  # Reuse same camera
    }
    
    metadata_path = episode_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def create_trajectory_h5(episode_dir, positions, yaws):
    """Create trajectory.h5 file from navigation data"""
    h5_path = episode_dir / "trajectory.h5"
    
    num_frames = len(positions)
    
    # Convert navigation data to DROID format
    # positions: [x, y] -> [x, y, z, rx, ry, rz]
    cartesian_positions = np.zeros((num_frames, 6), dtype=np.float64)
    cartesian_positions[:, 0] = positions[:, 0]  # x
    cartesian_positions[:, 1] = positions[:, 1]  # y
    cartesian_positions[:, 2] = 0.0  # z (fixed for 2D navigation)
    cartesian_positions[:, 3] = 0.0  # rx (roll)
    cartesian_positions[:, 4] = 0.0  # ry (pitch)
    cartesian_positions[:, 5] = yaws.flatten()  # rz (yaw)
    
    # Gripper position (not used in navigation, set to 0)
    gripper_positions = np.zeros(num_frames, dtype=np.float64)
    
    with h5py.File(h5_path, "w") as f:
        obs = f.create_group("observation")
        robot_state = obs.create_group("robot_state")
        
        # Save trajectory data
        robot_state.create_dataset("cartesian_position", data=cartesian_positions)
        robot_state.create_dataset("gripper_position", data=gripper_positions)
        
        # Create camera extrinsics (using zeros for fixed camera)
        camera_extrinsics = obs.create_group("camera_extrinsics")
        zero_extrinsics = np.zeros((num_frames, 6), dtype=np.float64)
        camera_extrinsics.create_dataset("nav_camera_left", data=zero_extrinsics)


def convert_episode(args):
    """Convert a single episode"""
    episode_path, output_dir, fps = args
    episode_name = os.path.basename(episode_path)
    
    try:
        # Load trajectory data
        traj_path = os.path.join(episode_path, 'traj_data.pkl')
        if not os.path.exists(traj_path):
            logger.warning(f"No traj_data.pkl found in {episode_path}, skipping")
            return None
        
        with open(traj_path, 'rb') as f:
            traj_data = pickle.load(f)
        
        positions = traj_data['position']
        yaws = traj_data['yaw']
        
        # Flatten yaws if needed
        if len(yaws.shape) > 1:
            yaws = yaws.flatten()
        
        # Check data consistency
        num_positions = len(positions)
        num_yaws = len(yaws)
        
        # Count images
        image_files = [f for f in os.listdir(episode_path) if f.endswith('.jpg')]
        num_images = len(image_files)
        
        if num_positions != num_yaws:
            logger.warning(f"Mismatch in {episode_name}: {num_positions} positions vs {num_yaws} yaws")
            return None
        
        if num_images != num_positions:
            logger.warning(f"Mismatch in {episode_name}: {num_images} images vs {num_positions} positions")
            # Use minimum to ensure consistency
            min_frames = min(num_images, num_positions)
            positions = positions[:min_frames]
            yaws = yaws[:min_frames]
        
        # Skip episodes that are too short
        if len(positions) < 8:  # V-JEPA expects at least 8 frames
            logger.warning(f"Episode {episode_name} too short ({len(positions)} frames), skipping")
            return None
        
        # Create output directory
        output_episode_dir = Path(output_dir) / episode_name
        output_episode_dir.mkdir(parents=True, exist_ok=True)
        
        # Create video directory
        video_dir = output_episode_dir / "recordings" / "MP4"
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Create video from images
        video_path = video_dir / "nav_camera.mp4"
        if not create_video_from_images(episode_path, video_path, fps):
            logger.error(f"Failed to create video for {episode_name}")
            return None
        
        # Create metadata.json
        create_metadata(output_episode_dir)
        
        # Create trajectory.h5
        create_trajectory_h5(output_episode_dir, positions, yaws)
        
        logger.info(f"Successfully converted {episode_name} ({len(positions)} frames)")
        return str(output_episode_dir)
        
    except Exception as e:
        logger.error(f"Error converting {episode_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_csv_file(output_dir, episode_paths):
    """Create CSV index file"""
    csv_path = Path(output_dir) / "go_stanford_train_paths.csv"
    
    with open(csv_path, "w") as f:
        for path in episode_paths:
            f.write(f"{path}\n")
    
    logger.info(f"Created CSV file: {csv_path} with {len(episode_paths)} episodes")


def main():
    parser = argparse.ArgumentParser(description="Convert go_stanford to V-JEPA2 format")
    parser.add_argument("--input_dir", type=str, 
                        default="/nvmessd/yinzi/navigation_datasets/go_stanford",
                        help="Path to go_stanford dataset")
    parser.add_argument("--output_dir", type=str, 
                        default="/nvmessd/yinzi/vjepa2/go_stanford_converted",
                        help="Output directory for converted data")
    parser.add_argument("--fps", type=int, default=10,
                        help="FPS for output videos")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Maximum number of episodes to convert (for testing)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all episode directories
    episode_dirs = []
    for item in sorted(os.listdir(args.input_dir)):
        item_path = os.path.join(args.input_dir, item)
        if os.path.isdir(item_path):
            # Check if it contains traj_data.pkl
            if os.path.exists(os.path.join(item_path, 'traj_data.pkl')):
                episode_dirs.append(item_path)
    
    logger.info(f"Found {len(episode_dirs)} episodes in {args.input_dir}")
    
    # Limit episodes if specified
    if args.max_episodes is not None:
        episode_dirs = episode_dirs[:args.max_episodes]
        logger.info(f"Limited to {len(episode_dirs)} episodes")
    
    # Prepare arguments for multiprocessing
    convert_args = [(ep, args.output_dir, args.fps) for ep in episode_dirs]
    
    # Convert episodes in parallel
    episode_paths = []
    
    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            with tqdm(total=len(convert_args), desc="Converting episodes") as pbar:
                for result in pool.imap_unordered(convert_episode, convert_args):
                    if result is not None:
                        episode_paths.append(result)
                    pbar.update(1)
    else:
        # Single-threaded for debugging
        for args_tuple in tqdm(convert_args, desc="Converting episodes"):
            result = convert_episode(args_tuple)
            if result is not None:
                episode_paths.append(result)
    
    # Create CSV file
    if episode_paths:
        create_csv_file(args.output_dir, sorted(episode_paths))
        logger.info(f"Successfully converted {len(episode_paths)} episodes")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"CSV file: {args.output_dir}/go_stanford_train_paths.csv")
    else:
        logger.error("No episodes were successfully converted")
    
    # Print summary statistics
    logger.info("\nConversion summary:")
    logger.info(f"Total episodes found: {len(episode_dirs)}")
    logger.info(f"Successfully converted: {len(episode_paths)}")
    logger.info(f"Failed: {len(episode_dirs) - len(episode_paths)}")


if __name__ == "__main__":
    main()