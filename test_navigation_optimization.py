#!/usr/bin/env python3
"""
Test script to verify navigation task optimization is working correctly.
"""

import torch
import yaml
from src.models.ac_predictor import VisionTransformerPredictorAC

def test_action_dimension_filtering():
    """Test if action dimension filtering works in AC predictor"""
    
    print("Testing Action Dimension Filtering for Navigation Task...")
    
    # Load configuration
    config_path = "configs/train/vitg16/go-stanford-finetune-8gpu.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    effective_action_dims = config['loss'].get('effective_action_dims', None)
    print(f"Effective action dims from config: {effective_action_dims}")
    
    # Create predictor with and without action filtering
    predictor_normal = VisionTransformerPredictorAC(
        img_size=(256, 256),
        patch_size=16,
        num_frames=2,
        tubelet_size=2,
        embed_dim=1408,  # vit_giant
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        effective_action_dims=None  # Normal mode
    )
    
    predictor_filtered = VisionTransformerPredictorAC(
        img_size=(256, 256),
        patch_size=16,
        num_frames=2,
        tubelet_size=2,
        embed_dim=1408,  # vit_giant
        predictor_embed_dim=1024,
        depth=24,
        num_heads=16,
        effective_action_dims=effective_action_dims  # Navigation optimized
    )
    
    # Create dummy inputs
    batch_size = 2
    tokens_per_frame = (256 // 16) ** 2  # 256 tokens per frame
    embed_dim = 1408
    
    x = torch.randn(batch_size, tokens_per_frame, embed_dim)  # Context tokens
    actions = torch.randn(batch_size, 1, 7)  # [x, y, z, rx, ry, rz, gripper]
    states = torch.randn(batch_size, 1, 7)   # Current states
    
    print("\nTesting forward pass...")
    
    # Test normal predictor
    with torch.no_grad():
        output_normal = predictor_normal(x, actions, states)
        print(f"Normal predictor output shape: {output_normal.shape}")
    
    # Test filtered predictor
    with torch.no_grad():
        output_filtered = predictor_filtered(x, actions, states)
        print(f"Filtered predictor output shape: {output_filtered.shape}")
    
    # Check if action mask is applied correctly
    if effective_action_dims is not None:
        action_mask = predictor_filtered.action_mask
        print(f"Action mask: {action_mask}")
        
        # Verify only effective dimensions have non-zero weights
        expected_mask = torch.zeros(7)
        expected_mask[effective_action_dims] = 1.0
        
        if torch.allclose(action_mask, expected_mask):
            print("✓ Action mask applied correctly")
        else:
            print("✗ Action mask not applied correctly")
            print(f"Expected: {expected_mask}")
            print(f"Got: {action_mask}")
    
    print("\nNavigation task optimization test completed!")

def test_config_loading():
    """Test if configuration is loaded correctly"""
    
    print("\nTesting Configuration Loading...")
    
    config_path = "configs/train/vitg16/go-stanford-finetune-8gpu.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    loss_config = config.get('loss', {})
    effective_action_dims = loss_config.get('effective_action_dims', None)
    action_loss_weight = loss_config.get('action_loss_weight', 1.0)
    
    print(f"✓ effective_action_dims: {effective_action_dims}")
    print(f"✓ action_loss_weight: {action_loss_weight}")
    
    if effective_action_dims == [0, 1, 5]:
        print("✓ Configuration loaded correctly for navigation task (x, y, yaw)")
    else:
        print("✗ Configuration not loaded correctly")

if __name__ == "__main__":
    test_config_loading()
    test_action_dimension_filtering()