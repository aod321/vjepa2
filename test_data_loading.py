#!/usr/bin/env python3
"""
Test script to verify DROID data loading for V-JEPA2
"""

import torch
import numpy as np
from app.vjepa_droid.droid import DROIDVideoDataset
from app.vjepa_droid.transforms import make_transforms

def test_data_loading():
    # Dataset parameters
    data_path = "/nvmessd/yinzi/vjepa2/droid_converted/droid_train_paths.csv"
    frames_per_clip = 8
    fps = 15
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
        print(f"  First state: {states[0]}")
        print(f"  First action: {actions[0]}")
        
        # Test multiple samples
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