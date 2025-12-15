
import h5py
import numpy as np

path = "/nvmessd/yinzi/vjepa2/datasets/ppo_dreamer_5000episodes_data_14-12-25-17_06/episode_0001_20251214_170620/trajectory.h5"

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: {obj.shape} {obj.dtype}")
    else:
        print(f"{name}/")

with h5py.File(path, 'r') as f:
    f.visititems(print_structure)
    
    print("\n--- Value Check ---")
    try:
        cp = f["observation/robot_state/cartesian_position"][:]
        gp = f["observation/robot_state/gripper_position"][:]
        print(f"cartesian_position shape: {cp.shape}")
        print(f"gripper_position shape: {gp.shape}")
        
    except Exception as e:
        print(f"Error accessing keys: {e}")
