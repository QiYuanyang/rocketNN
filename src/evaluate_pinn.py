"""
Evaluation script for trained Soft-PINN models.
Loads best models and evaluates on test set.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse

from train_pinn_model import SoftPINNModel, PhaseClassifier
from pinn_data_loader import TrajectoryDataset, create_data_loaders


def denormalize_data(data, mean, std):
    """Denormalize data using saved statistics."""
    return data * std + mean


@torch.no_grad()
def evaluate_models(models, test_loader, device, stats):
    """Evaluate models on test set and compute real-world metrics."""
    
    for model in models:
        model.eval()
    
    all_pos_errors = []
    all_vel_errors = []
    all_predictions = []
    all_truths = []
    
    for batch in test_loader:
        params = batch['params'].to(device)
        time = batch['time'].to(device)
        position_true = batch['position'].to(device)
        velocity_true = batch['velocity'].to(device)
        
        batch_size, n_points = time.shape
        
        # Get apogee times
        apogee_times = time[:, n_points // 2].cpu().numpy()
        
        # Classify phases
        phase_masks = []
        for b in range(batch_size):
            time_b = time[b].cpu().numpy()
            velocity_b = velocity_true[b].cpu().numpy()
            phases = PhaseClassifier.classify_phases(time_b, velocity_b, apogee_times[b])
            phase_masks.append(phases)
        phase_masks = np.array(phase_masks)
        
        # Forward pass
        position_pred = torch.zeros_like(position_true)
        velocity_pred = torch.zeros_like(velocity_true)
        
        for phase_idx, model in enumerate(models):
            mask = torch.from_numpy(phase_masks == phase_idx).to(device)
            if mask.sum() == 0:
                continue
            
            pos_pred, vel_pred = model(time, params)
            mask_expanded = mask.unsqueeze(-1).float()
            position_pred += pos_pred * mask_expanded
            velocity_pred += vel_pred * mask_expanded
        
        # Denormalize predictions and ground truth
        pos_true_denorm = denormalize_data(
            position_true.cpu().numpy(),
            np.array(stats['pos_mean']),
            np.array(stats['pos_std'])
        )
        pos_pred_denorm = denormalize_data(
            position_pred.cpu().numpy(),
            np.array(stats['pos_mean']),
            np.array(stats['pos_std'])
        )
        
        vel_true_denorm = denormalize_data(
            velocity_true.cpu().numpy(),
            np.array(stats['vel_mean']),
            np.array(stats['vel_std'])
        )
        vel_pred_denorm = denormalize_data(
            velocity_pred.cpu().numpy(),
            np.array(stats['vel_mean']),
            np.array(stats['vel_std'])
        )
        
        # Compute errors
        pos_errors = np.sqrt(np.sum((pos_pred_denorm - pos_true_denorm) ** 2, axis=-1))  # (batch, n_points)
        vel_errors = np.sqrt(np.sum((vel_pred_denorm - vel_true_denorm) ** 2, axis=-1))
        
        all_pos_errors.append(pos_errors)
        all_vel_errors.append(vel_errors)
        all_predictions.append({
            'position': pos_pred_denorm,
            'velocity': vel_pred_denorm,
            'time': time.cpu().numpy(),
            'params': params.cpu().numpy()
        })
        all_truths.append({
            'position': pos_true_denorm,
            'velocity': vel_true_denorm
        })
    
    # Concatenate all batches
    all_pos_errors = np.concatenate(all_pos_errors, axis=0)  # (n_test, n_points)
    all_vel_errors = np.concatenate(all_vel_errors, axis=0)
    
    # Compute statistics
    metrics = {
        'position_rmse_mean': np.sqrt(np.mean(all_pos_errors ** 2)),
        'position_rmse_std': np.std(np.sqrt(np.mean(all_pos_errors ** 2, axis=1))),
        'position_max_error': np.max(all_pos_errors),
        'velocity_rmse_mean': np.sqrt(np.mean(all_vel_errors ** 2)),
        'velocity_rmse_std': np.std(np.sqrt(np.mean(all_vel_errors ** 2, axis=1))),
        'velocity_max_error': np.max(all_vel_errors),
        'position_error_at_apogee': np.mean(all_pos_errors[:, len(all_pos_errors[0]) // 2]),
        'position_error_at_landing': np.mean(all_pos_errors[:, -1])
    }
    
    return metrics, all_predictions, all_truths, all_pos_errors, all_vel_errors


def plot_error_distribution(pos_errors, vel_errors, save_path):
    """Plot distribution of errors across test set."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Position RMSE per trajectory
    pos_rmse_per_traj = np.sqrt(np.mean(pos_errors ** 2, axis=1))
    axes[0, 0].hist(pos_rmse_per_traj, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(pos_rmse_per_traj), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(pos_rmse_per_traj):.2f}m')
    axes[0, 0].set_xlabel('Position RMSE per Trajectory (m)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Position Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Velocity RMSE per trajectory
    vel_rmse_per_traj = np.sqrt(np.mean(vel_errors ** 2, axis=1))
    axes[0, 1].hist(vel_rmse_per_traj, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(np.mean(vel_rmse_per_traj), color='r', linestyle='--',
                       label=f'Mean: {np.mean(vel_rmse_per_traj):.2f} m/s')
    axes[0, 1].set_xlabel('Velocity RMSE per Trajectory (m/s)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Velocity Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Position error over time
    mean_pos_error = np.mean(pos_errors, axis=0)
    std_pos_error = np.std(pos_errors, axis=0)
    time_points = np.arange(len(mean_pos_error))
    
    axes[1, 0].plot(time_points, mean_pos_error, 'b-', linewidth=2, label='Mean Error')
    axes[1, 0].fill_between(time_points, 
                            mean_pos_error - std_pos_error,
                            mean_pos_error + std_pos_error,
                            alpha=0.3, label='±1 std')
    axes[1, 0].set_xlabel('Time Point')
    axes[1, 0].set_ylabel('Position Error (m)')
    axes[1, 0].set_title('Position Error Over Flight')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Velocity error over time
    mean_vel_error = np.mean(vel_errors, axis=0)
    std_vel_error = np.std(vel_errors, axis=0)
    
    axes[1, 1].plot(time_points, mean_vel_error, 'g-', linewidth=2, label='Mean Error')
    axes[1, 1].fill_between(time_points,
                            mean_vel_error - std_vel_error,
                            mean_vel_error + std_vel_error,
                            alpha=0.3, label='±1 std')
    axes[1, 1].set_xlabel('Time Point')
    axes[1, 1].set_ylabel('Velocity Error (m/s)')
    axes[1, 1].set_title('Velocity Error Over Flight')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved error distribution plot to {save_path}")


def plot_sample_trajectories(predictions, truths, save_dir, n_samples=5):
    """Plot sample trajectory predictions in 3D."""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    n_test = len(predictions[0]['position'])
    indices = np.random.choice(n_test, min(n_samples, n_test), replace=False)
    
    for idx in indices:
        # Gather data for this trajectory
        pos_pred = predictions[0]['position'][idx]  # (n_points, 3)
        pos_true = truths[0]['position'][idx]
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 5))
        
        # 3D trajectory
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(pos_true[:, 0], pos_true[:, 1], pos_true[:, 2], 'b-', 
                linewidth=2, label='True', alpha=0.7)
        ax1.plot(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2], 'r--',
                linewidth=2, label='Predicted', alpha=0.7)
        ax1.scatter([0], [0], [0], c='g', s=100, marker='o', label='Launch')
        ax1.scatter([pos_true[-1, 0]], [pos_true[-1, 1]], [pos_true[-1, 2]], 
                   c='b', s=100, marker='x', label='True Landing')
        ax1.scatter([pos_pred[-1, 0]], [pos_pred[-1, 1]], [pos_pred[-1, 2]],
                   c='r', s=100, marker='x', label='Pred Landing')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # XY projection (top view)
        ax2 = fig.add_subplot(132)
        ax2.plot(pos_true[:, 0], pos_true[:, 1], 'b-', linewidth=2, label='True', alpha=0.7)
        ax2.plot(pos_pred[:, 0], pos_pred[:, 1], 'r--', linewidth=2, label='Predicted', alpha=0.7)
        ax2.scatter([0], [0], c='g', s=100, marker='o', label='Launch')
        ax2.scatter([pos_true[-1, 0]], [pos_true[-1, 1]], c='b', s=100, marker='x')
        ax2.scatter([pos_pred[-1, 0]], [pos_pred[-1, 1]], c='r', s=100, marker='x')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top View (XY Plane)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # Altitude vs time
        ax3 = fig.add_subplot(133)
        n_points = len(pos_true)
        time_points = np.arange(n_points)
        ax3.plot(time_points, pos_true[:, 2], 'b-', linewidth=2, label='True', alpha=0.7)
        ax3.plot(time_points, pos_pred[:, 2], 'r--', linewidth=2, label='Predicted', alpha=0.7)
        ax3.set_xlabel('Time Point')
        ax3.set_ylabel('Altitude (m)')
        ax3.set_title('Altitude Profile')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Calculate landing error
        landing_error = np.linalg.norm(pos_pred[-1] - pos_true[-1])
        apogee_true = np.max(pos_true[:, 2])
        apogee_pred = np.max(pos_pred[:, 2])
        apogee_error = abs(apogee_pred - apogee_true)
        
        plt.suptitle(f'Trajectory {idx} | Landing Error: {landing_error:.2f}m | Apogee Error: {apogee_error:.2f}m',
                    fontsize=14)
        plt.tight_layout()
        plt.savefig(save_dir / f'trajectory_3d_{idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved 3D trajectory plot for sample {idx}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Soft-PINN models')
    parser.add_argument('--data_dir', type=str, default='data/pinn_trajectories',
                       help='Directory containing trajectory data')
    parser.add_argument('--model_dir', type=str, default='models/pinn',
                       help='Directory containing trained models')
    parser.add_argument('--save_dir', type=str, default='results/pinn_evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load normalization stats
    stats_path = Path(args.data_dir) / 'normalization_stats.json'
    with open(stats_path) as f:
        stats = json.load(f)
    
    # Load data
    print("Loading test data...")
    _, _, test_loader, _ = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        train_split=0.7,
        val_split=0.15,
        num_workers=0
    )
    
    print(f"Test batches: {len(test_loader)}")
    
    # Load models
    print("Loading trained models...")
    models = []
    for phase_idx in range(3):
        model = SoftPINNModel(
            param_dim=10,
            time_encoding_dim=16,
            hidden_dim=256,
            n_res_blocks=3
        ).to(device)
        
        model_path = Path(args.model_dir) / f'best_model_phase{phase_idx}.pt'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models.append(model)
        print(f"✓ Loaded Phase {phase_idx} model from {model_path}")
    
    # Evaluate
    print("\nEvaluating models...")
    metrics, predictions, truths, pos_errors, vel_errors = evaluate_models(
        models, test_loader, device, stats
    )
    
    # Print metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS (Denormalized - Real World Units)")
    print("="*60)
    print(f"Position RMSE (mean ± std): {metrics['position_rmse_mean']:.2f} ± {metrics['position_rmse_std']:.2f} m")
    print(f"Position Max Error: {metrics['position_max_error']:.2f} m")
    print(f"Position Error at Apogee: {metrics['position_error_at_apogee']:.2f} m")
    print(f"Position Error at Landing: {metrics['position_error_at_landing']:.2f} m")
    print()
    print(f"Velocity RMSE (mean ± std): {metrics['velocity_rmse_mean']:.2f} ± {metrics['velocity_rmse_std']:.2f} m/s")
    print(f"Velocity Max Error: {metrics['velocity_max_error']:.2f} m/s")
    print("="*60)
    
    # Save metrics
    with open(save_dir / 'evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot error distributions
    print("\nGenerating visualizations...")
    plot_error_distribution(pos_errors, vel_errors, save_dir / 'error_distribution.png')
    
    # Plot sample trajectories
    plot_sample_trajectories(predictions, truths, save_dir / 'trajectories_3d', n_samples=5)
    
    print(f"\n✓ Evaluation complete! Results saved to {save_dir}")


if __name__ == '__main__':
    main()
