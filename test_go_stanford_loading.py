#!/usr/bin/env python3
"""
Test script to verify go_stanford data loading for V-JEPA2
"""

import torch
import numpy as np
from app.vjepa_droid.droid import DROIDVideoDataset
from app.vjepa_droid.transforms import make_transforms

def test_data_loading():
    # Dataset parameters
    data_path = "/nvmessd/yinzi/vjepa2/go_stanford_converted/go_stanford_train_paths.csv"
    frames_per_clip = 8
    fps = 10
    crop_size = 256
    
    # Create transform
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=[1.0, 1.0],
        random_resize_scale=[1.0, 1.0],
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )
    
    # Create dataset
    dataset = DROIDVideoDataset(
        data_path=data_path,
        frames_per_clip=frames_per_clip,
        transform=transform,
        fps=fps,
        camera_views=["left_mp4_path"],
        frameskip=2,
        camera_frame=False,
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    
    # Test loading first sample
    print("\nTesting first sample...")
    try:
        buffer, actions, states, extrinsics, indices = dataset[0]
        
        print(f"✓ Video buffer shape: {buffer.shape}")
        print(f"✓ Actions shape: {actions.shape}")
        print(f"✓ States shape: {states.shape}")
        print(f"✓ Extrinsics shape: {extrinsics.shape}")
        print(f"✓ Indices: {indices}")
        
        # Check data types
        print(f"\nData types:")
        print(f"  Buffer dtype: {buffer.dtype}")
        print(f"  Actions dtype: {actions.dtype}")
        print(f"  States dtype: {states.dtype}")
        
        # Show sample values
        print(f"\nSample values:")
        print(f"  First state (x, y, z, rx, ry, rz, gripper): {states[0]}")
        print(f"  Last state: {states[-1]}")
        print(f"  First action (dx, dy, dz, drx, dry, drz, dgripper): {actions[0]}")
        
        # Check navigation-specific values
        print(f"\nNavigation-specific checks:")
        print(f"  Z values (should be 0): {states[:, 2]}")
        print(f"  Roll values (should be 0): {states[:, 3]}")
        print(f"  Pitch values (should be 0): {states[:, 4]}")
        print(f"  Yaw values: {states[:, 5]}")
        print(f"  Gripper values (should be 0): {states[:, 6]}")
        
        # Test all samples
        print(f"\nTesting all {len(dataset)} samples...")
        for i in range(len(dataset)):
            try:
                _ = dataset[i]
                print(f"  Sample {i}: ✓")
            except Exception as e:
                print(f"  Sample {i}: ✗ Error: {e}")
        
        print("\n✅ Data loading test successful!")
        
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_loading()