import matplotlib.pyplot as plt
import json
from PIL import Image
import numpy as np

# Load metrics
with open('results/pinn_evaluation/evaluation_metrics.json', 'r') as f:
    metrics = json.load(f)

# Create figure showing key results
fig = plt.figure(figsize=(16, 10))

# Display sample predictions
sample_indices = [0, 1, 3, 4]
for i, idx in enumerate(sample_indices):
    img = Image.open(f'results/pinn_evaluation/trajectories_3d/trajectory_3d_{idx}.png')
    ax = plt.subplot(2, 2, i+1)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f'Test Sample {idx}', fontsize=14, fontweight='bold')

plt.suptitle('Soft-PINN Predictions (3 Phase Models with Smooth Transitions)', 
             fontsize=16, fontweight='bold', y=0.98)

# Add metrics text
metrics_text = f"""
RESULTS WITH SOFT PHASE TRANSITIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Position RMSE: {metrics['position_rmse_mean']/1000:.2f} ± {metrics['position_rmse_std']/1000:.2f} km
Position Max Error: {metrics['position_max_error']/1000:.2f} km
Error at Apogee: {metrics['position_error_at_apogee']:.0f} m
Error at Landing: {metrics['position_error_at_landing']/1000:.2f} km

Velocity RMSE: {metrics['velocity_rmse_mean']:.1f} ± {metrics['velocity_rmse_std']:.1f} m/s
Velocity Max Error: {metrics['velocity_max_error']:.1f} m/s

KEY IMPROVEMENTS:
✓ Eliminated step-like discontinuities
✓ Smooth phase transitions (sigmoid blending)
✓ 3 distinct physics models maintained
✓ Continuity loss enforces boundary smoothness
"""

fig.text(0.5, 0.02, metrics_text, ha='center', va='bottom', 
         fontsize=11, family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0.12, 1, 0.96])
plt.savefig('results/pinn_evaluation/summary.png', dpi=150, bbox_inches='tight')
print("✓ Saved summary to results/pinn_evaluation/summary.png")

# Show error distribution
fig2, axes = plt.subplots(1, 2, figsize=(12, 4))

img_error = Image.open('results/pinn_evaluation/error_distribution.png')
axes[0].imshow(img_error)
axes[0].axis('off')
axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')

# Show training visualization from final epoch
import os
viz_files = os.listdir('results/pinn_viz/epoch_140')
if viz_files:
    img_train = Image.open(f'results/pinn_viz/epoch_140/{viz_files[0]}')
    axes[1].imshow(img_train)
    axes[1].axis('off')
    axes[1].set_title('Training Sample (Epoch 140)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('results/pinn_evaluation/error_and_training.png', dpi=150, bbox_inches='tight')
print("✓ Saved error & training viz to results/pinn_evaluation/error_and_training.png")

plt.close('all')
