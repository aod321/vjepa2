#!/usr/bin/env python3
"""
Save torch.hub V-JEPA2 AC model as a checkpoint file for training
"""

import torch
import os

def save_hub_model_as_checkpoint():
    print("Loading V-JEPA2 AC model from torch.hub...")
    encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
    
    # Create checkpoint directory
    checkpoint_dir = "/nvmessd/yinzi/vjepa2/checkpoints/pretrained"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create a checkpoint in the format expected by train.py
    checkpoint = {
        'encoder': encoder.state_dict(),
        'target_encoder': encoder.state_dict(),  # target_encoder is a copy of encoder
        'predictor': predictor.state_dict(),
        'epoch': 0,
        'loss': 0.0,
        'batch_size': 1,
        'world_size': 1,
        'lr': 0.0
    }
    
    # Save the checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "vjepa2_ac_vitg.pt")
    print(f"Saving checkpoint to: {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)
    
    print("✅ Checkpoint saved successfully!")
    
    # Also save model info
    print("\nModel information:")
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Predictor parameters: {sum(p.numel() for p in predictor.parameters()):,}")
    
    # Save a config snippet
    config_snippet = f"""
# Add this to your training config file:
meta:
  pretrain_checkpoint: {checkpoint_path}
  load_predictor: true
  load_encoder: true
  context_encoder_key: encoder
  target_encoder_key: target_encoder
"""
    
    print("\n" + "="*50)
    print("Configuration snippet for training:")
    print(config_snippet)
    
    return checkpoint_path

if __name__ == "__main__":
    save_hub_model_as_checkpoint()