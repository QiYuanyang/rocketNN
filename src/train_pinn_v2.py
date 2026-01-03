"""
Physics-Informed Neural Network for Rocket Trajectory Prediction (v2)
======================================================================

Key improvements over v1:
- Single unified model (no phase splitting)
- Predicts ONLY position from (time, params)
- Velocity computed via autodiff: v = dr/dt
- Acceleration computed via autodiff: a = dv/dt
- Physics loss: ||a_pred - a_data||² enforces gravity, drag, thrust

This ensures the model learns physically realistic trajectories.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from pinn_data_loader import TrajectoryDataset, create_data_loaders, collate_trajectories


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time coordinate."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        Args:
            t: (batch_size, n_points) time values in [0, 1]
        Returns:
            encoding: (batch_size, n_points, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        
        # Create frequency bands
        freq = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=device) * 
            -(np.log(10000.0) / half_dim)
        )
        
        # Apply to time
        args = t.unsqueeze(-1) * freq.unsqueeze(0).unsqueeze(0)
        
        # Concatenate sin and cos
        encoding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return encoding


class ResidualBlock(nn.Module):
    """Residual block with layer normalization."""
    
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.activation = nn.SiLU()
        
    def forward(self, x):
        return self.activation(x + self.layers(x))


class PhysicsPINNModel(nn.Module):
    """
    Physics-Informed Neural Network that predicts ONLY position.
    Velocity and acceleration are computed via automatic differentiation.
    """
    
    def __init__(
        self,
        param_dim=10,
        time_encoding_dim=32,
        hidden_dim=256,
        n_res_blocks=3
    ):
        super().__init__()
        
        self.param_dim = param_dim
        self.time_encoding_dim = time_encoding_dim
        self.hidden_dim = hidden_dim
        
        # Time encoder
        self.pos_encoder = PositionalEncoding(time_encoding_dim)
        
        # Parameter encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        # Main network
        input_dim = time_encoding_dim + 128
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_res_blocks)
        ])
        
        # Output head - ONLY POSITION
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3)  # x, y, z
        )
        
    def forward(self, time, params):
        """
        Args:
            time: (batch_size, n_points) normalized time [0, 1]
            params: (batch_size, param_dim) rocket parameters
        Returns:
            position: (batch_size, n_points, 3)
        """
        batch_size, n_points = time.shape
        
        # Encode time
        time_enc = self.pos_encoder(time)  # (batch, n_points, time_encoding_dim)
        
        # Encode parameters
        param_enc = self.param_encoder(params)  # (batch, 128)
        param_enc = param_enc.unsqueeze(1).expand(-1, n_points, -1)
        
        # Combine encodings
        combined = torch.cat([time_enc, param_enc], dim=-1)
        
        # Encoder
        x = self.encoder(combined)
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Output position
        position = self.position_head(x)
        
        return position


def compute_derivatives(position, time):
    """
    Compute velocity and acceleration via automatic differentiation.
    
    Args:
        position: (batch_size, n_points, 3) predicted positions
        time: (batch_size, n_points) time values (requires_grad=True)
        
    Returns:
        velocity: (batch_size, n_points, 3) dr/dt
        acceleration: (batch_size, n_points, 3) dv/dt
    """
    batch_size, n_points, _ = position.shape
    
    # Compute velocity: v = dr/dt
    velocity = []
    for i in range(3):  # x, y, z components
        pos_i = position[:, :, i]
        
        grad_outputs = torch.ones_like(pos_i)
        v_i = torch.autograd.grad(
            outputs=pos_i,
            inputs=time,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        velocity.append(v_i.unsqueeze(-1))
    
    velocity = torch.cat(velocity, dim=-1)  # (batch, n_points, 3)
    
    # Compute acceleration: a = dv/dt
    acceleration = []
    for i in range(3):
        vel_i = velocity[:, :, i]
        
        grad_outputs = torch.ones_like(vel_i)
        a_i = torch.autograd.grad(
            outputs=vel_i,
            inputs=time,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        acceleration.append(a_i.unsqueeze(-1))
    
    acceleration = torch.cat(acceleration, dim=-1)  # (batch, n_points, 3)
    
    return velocity, acceleration


def compute_initial_condition_loss(position, velocity):
    """
    Enforce initial conditions: r(0) = 0, v(0) = 0
    """
    pos_ic_loss = (position[:, 0, :] ** 2).mean()
    vel_ic_loss = (velocity[:, 0, :] ** 2).mean()
    return pos_ic_loss + vel_ic_loss


def train_epoch(model, train_loader, optimizer, device, lambda_accel=10.0, lambda_ic=10.0):
    """Train for one epoch with acceleration physics loss."""
    
    model.train()
    
    total_loss = 0.0
    total_pos_loss = 0.0
    total_accel_loss = 0.0
    total_ic_loss = 0.0
    n_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        params = batch['params'].to(device)
        time = batch['time'].to(device)
        position_true = batch['position'].to(device)
        velocity_true = batch['velocity'].to(device)
        acceleration_true = batch['acceleration'].to(device)
        
        # Enable gradient for time (needed for autodiff)
        time.requires_grad_(True)
        
        # Forward pass
        position_pred = model(time, params)
        
        # Compute derivatives via autodiff
        velocity_pred, acceleration_pred = compute_derivatives(position_pred, time)
        
        # Position loss
        pos_loss = nn.functional.mse_loss(position_pred, position_true)
        
        # PHYSICS LOSS: Match predicted acceleration to true acceleration
        # This enforces gravity, drag, thrust - all forces!
        accel_loss = nn.functional.mse_loss(acceleration_pred, acceleration_true)
        
        # Initial condition loss
        ic_loss = compute_initial_condition_loss(position_pred, velocity_pred)
        
        # Total loss
        loss = pos_loss + lambda_accel * accel_loss + lambda_ic * ic_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_pos_loss += pos_loss.item()
        total_accel_loss += accel_loss.item()
        total_ic_loss += ic_loss.item()
        n_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'pos': f'{pos_loss.item():.4f}',
            'accel': f'{accel_loss.item():.4f}',
            'ic': f'{ic_loss.item():.4f}'
        })
    
    return {
        'loss': total_loss / n_batches,
        'pos_loss': total_pos_loss / n_batches,
        'accel_loss': total_accel_loss / n_batches,
        'ic_loss': total_ic_loss / n_batches
    }


def validate_epoch(model, val_loader, device):
    """Validate the model."""
    
    model.eval()
    
    total_pos_rmse = 0.0
    total_vel_rmse = 0.0
    n_batches = 0
    
    # We need gradients for time, but don't want to update model params
    for batch in val_loader:
        params = batch['params'].to(device)
        time = batch['time'].to(device)
        position_true = batch['position'].to(device)
        velocity_true = batch['velocity'].to(device)
        
        # Enable gradient for time
        time.requires_grad_(True)
        
        # Forward pass - still compute gradients but don't update params
        position_pred = model(time, params)
        
        # Compute velocity via autodiff
        velocity_pred = []
        for i in range(3):
            pos_i = position_pred[:, :, i]
            grad_outputs = torch.ones_like(pos_i)
            v_i = torch.autograd.grad(
                outputs=pos_i,
                inputs=time,
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=True if i < 2 else False
            )[0]
            velocity_pred.append(v_i.unsqueeze(-1))
        
        velocity_pred = torch.cat(velocity_pred, dim=-1)
        
        # Compute RMSE (detach to avoid memory buildup)
        with torch.no_grad():
            pos_rmse = torch.sqrt(((position_pred - position_true) ** 2).mean())
            vel_rmse = torch.sqrt(((velocity_pred - velocity_true) ** 2).mean())
        
        total_pos_rmse += pos_rmse.item()
        total_vel_rmse += vel_rmse.item()
        n_batches += 1
    
    return {
        'position_rmse': total_pos_rmse / n_batches,
        'velocity_rmse': total_vel_rmse / n_batches
    }


def visualize_predictions(model, dataset, device, save_dir, n_samples=3):
    """Visualize model predictions."""
    
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        params = sample['params'].unsqueeze(0).to(device)
        time = sample['time'].unsqueeze(0).to(device)
        position_true = sample['position'].unsqueeze(0).to(device)
        velocity_true = sample['velocity'].unsqueeze(0).to(device)
        acceleration_true = sample['acceleration'].unsqueeze(0).to(device)
        
        # Enable gradient for time
        time.requires_grad_(True)
        
        # Forward pass (need gradients for derivatives, but don't update model)
        position_pred = model(time, params)
        velocity_pred, acceleration_pred = compute_derivatives(position_pred, time)
        
        # Convert to numpy (detach after computing derivatives)
        pos_true_np = position_true[0].cpu().numpy()
        pos_pred_np = position_pred[0].detach().cpu().numpy()
        vel_true_np = velocity_true[0].cpu().numpy()
        vel_pred_np = velocity_pred[0].detach().cpu().numpy()
        acc_true_np = acceleration_true[0].cpu().numpy()
        acc_pred_np = acceleration_pred[0].detach().cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        
        # Position plots
        for i, label in enumerate(['X', 'Y', 'Z (Altitude)']):
            axes[0, i].plot(pos_true_np[:, i], label='True', linewidth=2)
            axes[0, i].plot(pos_pred_np[:, i], label='Predicted', linestyle='--', linewidth=2)
            axes[0, i].set_title(f'Position {label}')
            axes[0, i].set_xlabel('Time Point')
            axes[0, i].set_ylabel('Position (m)')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
        
        # Velocity plots
        for i, label in enumerate(['Vx', 'Vy', 'Vz']):
            axes[1, i].plot(vel_true_np[:, i], label='True', linewidth=2)
            axes[1, i].plot(vel_pred_np[:, i], label='Predicted', linestyle='--', linewidth=2)
            axes[1, i].set_title(f'Velocity {label}')
            axes[1, i].set_xlabel('Time Point')
            axes[1, i].set_ylabel('Velocity (m/s)')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        # Acceleration plots
        for i, label in enumerate(['Ax', 'Ay', 'Az']):
            axes[2, i].plot(acc_true_np[:, i], label='True', linewidth=2)
            axes[2, i].plot(acc_pred_np[:, i], label='Predicted', linestyle='--', linewidth=2)
            axes[2, i].set_title(f'Acceleration {label}')
            axes[2, i].set_xlabel('Time Point')
            axes[2, i].set_ylabel('Acceleration (m/s²)')
            axes[2, i].legend()
            axes[2, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'prediction_{idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization for trajectory {idx}")


def main():
    parser = argparse.ArgumentParser(description='Train Physics-Informed PINN v2')
    parser.add_argument('--data_dir', type=str, default='data/pinn_trajectories')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_accel', type=float, default=10.0,
                       help='Weight for acceleration physics loss')
    parser.add_argument('--lambda_ic', type=float, default=10.0)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_res_blocks', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default='models/pinn_v2')
    parser.add_argument('--viz_dir', type=str, default='results/pinn_v2_viz')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, dataset = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating Physics-PINN model...")
    model = PhysicsPINNModel(
        param_dim=10,
        time_encoding_dim=32,
        hidden_dim=args.hidden_dim,
        n_res_blocks=args.n_res_blocks
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'train_pos_loss': [],
        'train_accel_loss': [],
        'val_position_rmse': [],
        'val_velocity_rmse': []
    }
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            lambda_accel=args.lambda_accel,
            lambda_ic=args.lambda_ic
        )
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_pos_loss'].append(train_metrics['pos_loss'])
        history['train_accel_loss'].append(train_metrics['accel_loss'])
        history['val_position_rmse'].append(val_metrics['position_rmse'])
        history['val_velocity_rmse'].append(val_metrics['velocity_rmse'])
        
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Pos RMSE: {val_metrics['position_rmse']:.4f} | "
              f"Val Vel RMSE: {val_metrics['velocity_rmse']:.4f}")
        
        # Save best model
        val_loss = val_metrics['position_rmse'] + val_metrics['velocity_rmse']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / 'best_model.pt')
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Visualize predictions
        if (epoch + 1) % 20 == 0:
            viz_dir = Path(args.viz_dir) / f'epoch_{epoch+1}'
            visualize_predictions(model, dataset, device, viz_dir, n_samples=2)
    
    # Save training history
    save_dir = Path(args.save_dir)
    with open(save_dir / 'training_history_v2.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history['train_loss'])
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['train_pos_loss'], label='Position')
    axes[0, 1].plot(history['train_accel_loss'], label='Acceleration')
    axes[0, 1].set_title('Training Loss Components')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history['val_position_rmse'])
    axes[1, 0].set_title('Validation Position RMSE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE (normalized)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(history['val_velocity_rmse'])
    axes[1, 1].set_title('Validation Velocity RMSE')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('RMSE (normalized)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves_v2.png', dpi=150)
    print(f"\n✓ Training complete!")
    print(f"Models saved to: {args.save_dir}")
    print(f"Visualizations saved to: {args.viz_dir}")


if __name__ == '__main__':
    main()
