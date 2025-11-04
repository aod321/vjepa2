import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Set the checkpoint directory
checkpoint_dir = Path("/nvmessd/yinzi/vjepa2/checkpoints/go_stanford_finetune_8gpu_0818_12_18")

# Read all log files
log_files = sorted([f for f in checkpoint_dir.glob("log_r*.csv")])
print(f"Found {len(log_files)} log files")

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot individual GPU losses
all_data = []
for log_file in log_files:
    df = pd.read_csv(log_file, skiprows=1)  # Skip the duplicate header
    rank = int(log_file.stem.split('_r')[1])
    
    # Ensure loss column is numeric
    df['loss'] = pd.to_numeric(df['loss'], errors='coerce')
    
    # Drop any rows with NaN loss values
    df = df.dropna(subset=['loss'])
    
    # Calculate global iteration number
    df['global_iter'] = df.index
    
    # Plot individual GPU loss
    ax1.plot(df['global_iter'], df['loss'], alpha=0.7, linewidth=1, label=f'GPU {rank}')
    
    all_data.append(df)

# Calculate average loss across all GPUs
# Align data by iteration
max_iters = max(len(df) for df in all_data)
avg_losses = []
std_losses = []
valid_iters = []

for i in range(max_iters):
    losses_at_iter = []
    for df in all_data:
        if i < len(df):
            losses_at_iter.append(df.iloc[i]['loss'])
    
    if losses_at_iter:
        avg_losses.append(np.mean(losses_at_iter))
        std_losses.append(np.std(losses_at_iter))
        valid_iters.append(i)

# Convert to numpy arrays
avg_losses = np.array(avg_losses)
std_losses = np.array(std_losses)
valid_iters = np.array(valid_iters)

# Plot average loss with error band
ax2.plot(valid_iters, avg_losses, 'b-', linewidth=2, label='Average')
ax2.fill_between(valid_iters, 
                 avg_losses - std_losses, 
                 avg_losses + std_losses, 
                 alpha=0.3, color='blue', label='±1 std')

# Customize first subplot
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss per GPU')
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

# Customize second subplot
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss')
ax2.set_title('Average Training Loss (8 GPUs)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)

# Add overall title
fig.suptitle('V-JEPA Fine-tuning Loss Curves - go_stanford_finetune_8gpu_0818_12_18', fontsize=14)
plt.tight_layout()

# Save the plot
output_path = checkpoint_dir / 'training_loss_curves.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to: {output_path}")

# Also save a zoomed-in version focusing on convergence
fig2, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(valid_iters, avg_losses, 'b-', linewidth=2, label='Average')
ax3.fill_between(valid_iters, 
                 avg_losses - std_losses, 
                 avg_losses + std_losses, 
                 alpha=0.3, color='blue', label='±1 std')

# Zoom in to see convergence better (last 80% of training)
start_idx = int(len(valid_iters) * 0.2)
ax3.set_xlim(valid_iters[start_idx], valid_iters[-1])
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Loss')
ax3.set_title('V-JEPA Fine-tuning Loss - Convergence View')
ax3.legend()
ax3.grid(True, alpha=0.3)

output_path2 = checkpoint_dir / 'training_loss_convergence.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"Saved convergence plot to: {output_path2}")

# Print some statistics
final_avg_loss = avg_losses[-1]
initial_loss = avg_losses[0]
print(f"\nTraining Statistics:")
print(f"Initial loss: {initial_loss:.4f}")
print(f"Final loss: {final_avg_loss:.4f}")
print(f"Loss reduction: {(initial_loss - final_avg_loss)/initial_loss * 100:.1f}%")
print(f"Total iterations: {len(valid_iters)}")

plt.show()