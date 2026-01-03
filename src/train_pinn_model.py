"""
Training script for Soft-PINN rocket trajectory prediction with phase-specific models.

Three models for three flight phases:
1. Powered Ascent: Launch to motor burnout
2. Coasting Ascent: Burnout to apogee
3. Descent: Apogee to landing
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
    
    def __init__(self, d_model=16, max_freq=10.0):
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq
        
    def forward(self, t):
        """
        Args:
            t: (batch_size, n_points) normalized time values
        Returns:
            (batch_size, n_points, d_model) encoded time
        """
        batch_size, n_points = t.shape
        
        # Create frequency bands
        freqs = torch.linspace(0, self.max_freq, self.d_model // 2, device=t.device)
        
        # Expand dimensions for broadcasting
        t_expanded = t.unsqueeze(-1)  # (batch, n_points, 1)
        freqs_expanded = freqs.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model//2)
        
        # Compute sin and cos
        angles = 2 * np.pi * freqs_expanded * t_expanded
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)
        
        # Concatenate sin and cos
        encoding = torch.cat([sin_enc, cos_enc], dim=-1)  # (batch, n_points, d_model)
        
        return encoding


class ResidualBlock(nn.Module):
    """Residual block with layer normalization."""
    
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x + residual
        return self.norm2(x)


class SoftPINNModel(nn.Module):
    """Soft-PINN model for one flight phase."""
    
    def __init__(self, param_dim=10, time_encoding_dim=16, hidden_dim=256, 
                 n_res_blocks=3, dropout=0.1):
        super().__init__()
        
        self.param_dim = param_dim
        self.time_encoding_dim = time_encoding_dim
        self.hidden_dim = hidden_dim
        
        # Positional encoding for time
        self.pos_encoder = PositionalEncoding(d_model=time_encoding_dim)
        
        # Parameter encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU()
        )
        
        # Combined encoder
        input_dim = time_encoding_dim + 128  # time encoding + param encoding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_res_blocks)
        ])
        
        # Output heads
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3)  # x, y, z
        )
        
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3)  # vx, vy, vz
        )
        
    def forward(self, time, params):
        """
        Args:
            time: (batch_size, n_points) normalized time
            params: (batch_size, param_dim) rocket parameters
        Returns:
            position: (batch_size, n_points, 3)
            velocity: (batch_size, n_points, 3)
        """
        batch_size, n_points = time.shape
        
        # Encode time
        time_enc = self.pos_encoder(time)  # (batch, n_points, time_encoding_dim)
        
        # Encode parameters
        param_enc = self.param_encoder(params)  # (batch, 128)
        param_enc = param_enc.unsqueeze(1).expand(-1, n_points, -1)  # (batch, n_points, 128)
        
        # Combine encodings
        combined = torch.cat([time_enc, param_enc], dim=-1)  # (batch, n_points, input_dim)
        
        # Encoder
        x = self.encoder(combined)
        
        # Residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Output heads
        position = self.position_head(x)
        velocity = self.velocity_head(x)
        
        return position, velocity


class PhaseClassifier:
    """Classify flight phases based on time and trajectory data."""
    
    @staticmethod
    def classify_phases(time, velocity, apogee_time):
        """
        Classify each time point into one of three phases.
        
        Args:
            time: (n_points,) array of time values
            velocity: (n_points, 3) array of velocities
            apogee_time: scalar apogee time
            
        Returns:
            phases: (n_points,) array of phase indices (0, 1, or 2)
        """
        n_points = len(time)
        phases = np.zeros(n_points, dtype=int)
        
        # Compute vertical velocity
        vz = velocity[:, 2] if len(velocity.shape) > 1 else velocity
        
        # Phase 0: Powered ascent (first 20% of time to apogee, or until vz peaks)
        apogee_idx = np.argmin(np.abs(time - apogee_time))
        max_vz_idx = np.argmax(vz[:apogee_idx+1])
        burnout_idx = min(int(apogee_idx * 0.2), max_vz_idx)
        
        # Phase 1: Coasting ascent (burnout to apogee)
        # Phase 2: Descent (after apogee)
        phases[:burnout_idx] = 0  # Powered ascent
        phases[burnout_idx:apogee_idx] = 1  # Coasting ascent
        phases[apogee_idx:] = 2  # Descent
        
        return phases


def compute_physics_loss(model, time, params, position_pred, velocity_pred, phase_mask=None):
    """
    Compute kinematic physics loss: ||∂r/∂t - v||²
    
    Args:
        model: PINN model (not used, but kept for API consistency)
        time: (batch_size, n_points) normalized time (requires_grad=True)
        params: (batch_size, param_dim) parameters
        position_pred: (batch_size, n_points, 3) predicted positions
        velocity_pred: (batch_size, n_points, 3) predicted velocities
        phase_mask: (batch_size, n_points) boolean mask for this phase
        
    Returns:
        physics_loss: scalar loss value
    """
    batch_size, n_points, _ = position_pred.shape
    
    # Compute gradients for each component
    dr_dt = []
    for i in range(3):  # x, y, z
        pos_component = position_pred[:, :, i]  # (batch, n_points)
        
        # Compute gradient w.r.t. time
        grad_outputs = torch.ones_like(pos_component)
        grads = torch.autograd.grad(
            outputs=pos_component,
            inputs=time,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]  # (batch, n_points)
        
        dr_dt.append(grads.unsqueeze(-1))
    
    dr_dt = torch.cat(dr_dt, dim=-1)  # (batch, n_points, 3)
    
    # Kinematic constraint: dr/dt = v
    residual = dr_dt - velocity_pred
    
    # Apply phase mask if provided
    if phase_mask is not None:
        mask = phase_mask.unsqueeze(-1).float()  # (batch, n_points, 1)
        residual = residual * mask
        n_masked = mask.sum()
        if n_masked > 0:
            physics_loss = (residual ** 2).sum() / n_masked
        else:
            physics_loss = torch.tensor(0.0, device=residual.device)
    else:
        physics_loss = (residual ** 2).mean()
    
    return physics_loss


def compute_initial_condition_loss(position_pred, velocity_pred):
    """
    Enforce initial conditions: r(0) = 0, v(0) = 0
    
    Args:
        position_pred: (batch_size, n_points, 3)
        velocity_pred: (batch_size, n_points, 3)
        
    Returns:
        ic_loss: scalar loss
    """
    # Initial position and velocity (first time point)
    r0 = position_pred[:, 0, :]  # (batch, 3)
    v0 = velocity_pred[:, 0, :]  # (batch, 3)
    
    ic_loss = (r0 ** 2).mean() + (v0 ** 2).mean()
    
    return ic_loss


def train_epoch(models, train_loader, optimizers, device, lambda_physics=0.1, lambda_ic=10.0):
    """Train for one epoch with phase-specific models."""
    
    for model in models:
        model.train()
    
    total_loss = 0.0
    total_data_loss = 0.0
    total_physics_loss = 0.0
    total_ic_loss = 0.0
    n_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch in progress_bar:
        params = batch['params'].to(device)  # (batch, param_dim)
        time = batch['time'].to(device)  # (batch, n_points)
        position_true = batch['position'].to(device)  # (batch, n_points, 3)
        velocity_true = batch['velocity'].to(device)  # (batch, n_points, 3)
        
        batch_size, n_points = time.shape
        
        # Require gradient for time (for physics loss)
        time.requires_grad_(True)
        
        # Get apogee times for phase classification
        # Assume apogee is around middle of trajectory
        apogee_times = time[:, n_points // 2].detach().cpu().numpy()
        
        # Classify phases for each trajectory
        phase_masks = []
        for b in range(batch_size):
            time_b = time[b].detach().cpu().numpy()
            velocity_b = velocity_true[b].detach().cpu().numpy()
            phases = PhaseClassifier.classify_phases(time_b, velocity_b, apogee_times[b])
            phase_masks.append(phases)
        phase_masks = np.array(phase_masks)  # (batch, n_points)
        
        # Forward pass for each phase model
        position_preds = []
        velocity_preds = []
        
        for phase_idx, model in enumerate(models):
            # Get mask for this phase
            mask = torch.from_numpy(phase_masks == phase_idx).to(device)  # (batch, n_points)
            
            if mask.sum() == 0:  # Skip if no points in this phase
                continue
            
            # Forward pass
            pos_pred, vel_pred = model(time, params)
            position_preds.append((pos_pred, mask))
            velocity_preds.append((vel_pred, mask))
        
        # Combine predictions from all phases
        position_pred = torch.zeros_like(position_true)
        velocity_pred = torch.zeros_like(velocity_true)
        
        for (pos, mask), (vel, _) in zip(position_preds, velocity_preds):
            mask_expanded = mask.unsqueeze(-1).float()
            position_pred += pos * mask_expanded
            velocity_pred += vel * mask_expanded
        
        # Data loss (MSE)
        data_loss_pos = nn.functional.mse_loss(position_pred, position_true)
        data_loss_vel = nn.functional.mse_loss(velocity_pred, velocity_true)
        data_loss = data_loss_pos + data_loss_vel
        
        # Physics loss for each phase
        physics_loss = 0.0
        for phase_idx, model in enumerate(models):
            mask = torch.from_numpy(phase_masks == phase_idx).to(device)
            if mask.sum() > 0:
                phase_physics_loss = compute_physics_loss(
                    model, time, params, position_pred, velocity_pred, phase_mask=mask
                )
                physics_loss += phase_physics_loss
        
        # Initial condition loss (only for phase 0 model)
        pos_phase0, vel_phase0 = models[0](time, params)
        ic_loss = compute_initial_condition_loss(pos_phase0, vel_phase0)
        
        # Total loss
        loss = data_loss + lambda_physics * physics_loss + lambda_ic * ic_loss
        
        # Backward pass
        for optimizer in optimizers:
            optimizer.zero_grad()
        
        loss.backward()
        
        # Gradient clipping
        for model in models:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        for optimizer in optimizers:
            optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_data_loss += data_loss.item()
        total_physics_loss += physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss
        total_ic_loss += ic_loss.item()
        n_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'data': f'{data_loss.item():.4f}',
            'physics': f'{physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss:.4f}',
            'ic': f'{ic_loss.item():.4f}'
        })
    
    return {
        'loss': total_loss / n_batches,
        'data_loss': total_data_loss / n_batches,
        'physics_loss': total_physics_loss / n_batches,
        'ic_loss': total_ic_loss / n_batches
    }


@torch.no_grad()
def validate_epoch(models, val_loader, device):
    """Validate the models."""
    
    for model in models:
        model.eval()
    
    total_loss_pos = 0.0
    total_loss_vel = 0.0
    n_batches = 0
    
    for batch in val_loader:
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
        
        # Forward pass for each phase
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
        
        # Compute losses
        loss_pos = nn.functional.mse_loss(position_pred, position_true)
        loss_vel = nn.functional.mse_loss(velocity_pred, velocity_true)
        
        total_loss_pos += loss_pos.item()
        total_loss_vel += loss_vel.item()
        n_batches += 1
    
    return {
        'position_mse': total_loss_pos / n_batches,
        'velocity_mse': total_loss_vel / n_batches,
        'position_rmse': np.sqrt(total_loss_pos / n_batches),
        'velocity_rmse': np.sqrt(total_loss_vel / n_batches)
    }


def visualize_predictions(models, dataset, device, save_dir, n_samples=3):
    """Visualize model predictions on sample trajectories."""
    
    for model in models:
        model.eval()
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Select random samples
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        params = sample['params'].unsqueeze(0).to(device)  # (1, param_dim)
        time = sample['time'].unsqueeze(0).to(device)  # (1, n_points)
        position_true = sample['position'].unsqueeze(0).to(device)  # (1, n_points, 3)
        velocity_true = sample['velocity'].unsqueeze(0).to(device)  # (1, n_points, 3)
        
        # Classify phases
        time_np = time[0].cpu().numpy()
        velocity_np = velocity_true[0].cpu().numpy()
        apogee_time = time_np[len(time_np) // 2]
        phases = PhaseClassifier.classify_phases(time_np, velocity_np, apogee_time)
        
        # Forward pass
        with torch.no_grad():
            position_pred = torch.zeros_like(position_true)
            velocity_pred = torch.zeros_like(velocity_true)
            
            for phase_idx, model in enumerate(models):
                mask = torch.from_numpy(phases == phase_idx).to(device)
                if mask.sum() == 0:
                    continue
                
                pos_pred, vel_pred = model(time, params)
                mask_expanded = mask.unsqueeze(0).unsqueeze(-1).float()
                position_pred += pos_pred * mask_expanded
                velocity_pred += vel_pred * mask_expanded
        
        # Convert to numpy
        time_np = time[0].cpu().numpy()
        pos_true_np = position_true[0].cpu().numpy()
        pos_pred_np = position_pred[0].cpu().numpy()
        vel_true_np = velocity_true[0].cpu().numpy()
        vel_pred_np = velocity_pred[0].cpu().numpy()
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Position plots
        for i, (label, color) in enumerate(zip(['X', 'Y', 'Z'], ['r', 'g', 'b'])):
            ax = axes[0, i]
            ax.plot(time_np, pos_true_np[:, i], f'{color}-', label='True', linewidth=2)
            ax.plot(time_np, pos_pred_np[:, i], f'{color}--', label='Predicted', linewidth=2)
            
            # Color by phase
            for phase_idx in range(3):
                mask = phases == phase_idx
                if mask.sum() > 0:
                    ax.axvspan(time_np[mask].min(), time_np[mask].max(), 
                              alpha=0.2, color=f'C{phase_idx}')
            
            ax.set_xlabel('Time (normalized)')
            ax.set_ylabel(f'{label} Position')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Velocity plots
        for i, (label, color) in enumerate(zip(['Vx', 'Vy', 'Vz'], ['r', 'g', 'b'])):
            ax = axes[1, i]
            ax.plot(time_np, vel_true_np[:, i], f'{color}-', label='True', linewidth=2)
            ax.plot(time_np, vel_pred_np[:, i], f'{color}--', label='Predicted', linewidth=2)
            
            # Color by phase
            for phase_idx in range(3):
                mask = phases == phase_idx
                if mask.sum() > 0:
                    ax.axvspan(time_np[mask].min(), time_np[mask].max(), 
                              alpha=0.2, color=f'C{phase_idx}')
            
            ax.set_xlabel('Time (normalized)')
            ax.set_ylabel(f'{label} Velocity')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Trajectory {idx} - Phase Colors: Phase0(Blue), Phase1(Orange), Phase2(Green)', 
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(save_dir / f'prediction_{idx}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization for trajectory {idx}")


def main():
    parser = argparse.ArgumentParser(description='Train Soft-PINN models for rocket trajectory prediction')
    parser.add_argument('--data_dir', type=str, default='data/pinn_trajectories',
                       help='Directory containing trajectory data')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--lambda_physics', type=float, default=0.1,
                       help='Weight for physics loss')
    parser.add_argument('--lambda_ic', type=float, default=10.0,
                       help='Weight for initial condition loss')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--n_res_blocks', type=int, default=3,
                       help='Number of residual blocks')
    parser.add_argument('--save_dir', type=str, default='models/pinn',
                       help='Directory to save models')
    parser.add_argument('--viz_dir', type=str, default='results/pinn_viz',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = Path(args.viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, train_dataset = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        train_split=0.7,
        val_split=0.15,
        num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Create 3 phase-specific models
    print("Creating models...")
    models = [
        SoftPINNModel(
            param_dim=10,
            time_encoding_dim=16,
            hidden_dim=args.hidden_dim,
            n_res_blocks=args.n_res_blocks
        ).to(device) for _ in range(3)
    ]
    
    print(f"Phase 0 (Powered Ascent) model: {sum(p.numel() for p in models[0].parameters()):,} parameters")
    print(f"Phase 1 (Coasting Ascent) model: {sum(p.numel() for p in models[1].parameters()):,} parameters")
    print(f"Phase 2 (Descent) model: {sum(p.numel() for p in models[2].parameters()):,} parameters")
    
    # Optimizers for each model
    optimizers = [
        optim.Adam(model.parameters(), lr=args.lr) for model in models
    ]
    
    # Learning rate schedulers
    schedulers = [
        optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
        for opt in optimizers
    ]
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'train_data_loss': [],
        'train_physics_loss': [],
        'train_ic_loss': [],
        'val_position_rmse': [],
        'val_velocity_rmse': []
    }
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            models, train_loader, optimizers, device,
            lambda_physics=args.lambda_physics,
            lambda_ic=args.lambda_ic
        )
        
        # Validate
        val_metrics = validate_epoch(models, val_loader, device)
        
        # Update learning rate
        for scheduler in schedulers:
            scheduler.step()
        
        # Log metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_data_loss'].append(train_metrics['data_loss'])
        history['train_physics_loss'].append(train_metrics['physics_loss'])
        history['train_ic_loss'].append(train_metrics['ic_loss'])
        history['val_position_rmse'].append(val_metrics['position_rmse'])
        history['val_velocity_rmse'].append(val_metrics['velocity_rmse'])
        
        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"Val Pos RMSE: {val_metrics['position_rmse']:.4f} | "
              f"Val Vel RMSE: {val_metrics['velocity_rmse']:.4f}")
        
        # Save best model
        val_loss = val_metrics['position_rmse'] + val_metrics['velocity_rmse']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            for phase_idx, model in enumerate(models):
                torch.save(model.state_dict(), save_dir / f'best_model_phase{phase_idx}.pt')
            print(f"✓ Saved best models (val_loss: {val_loss:.4f})")
        
        # Periodic visualization
        if (epoch + 1) % 20 == 0 or epoch == 0:
            visualize_predictions(models, train_dataset, device, viz_dir / f'epoch_{epoch+1}', n_samples=2)
    
    # Save final models
    for phase_idx, model in enumerate(models):
        torch.save(model.state_dict(), save_dir / f'final_model_phase{phase_idx}.pt')
    
    # Save training history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['train_data_loss'], label='Data Loss')
    axes[0, 1].plot(history['train_physics_loss'], label='Physics Loss')
    axes[0, 1].plot(history['train_ic_loss'], label='IC Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history['val_position_rmse'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('Validation Position RMSE')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(history['val_velocity_rmse'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].set_title('Validation Velocity RMSE')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n✓ Training complete!")
    print(f"Models saved to: {save_dir}")
    print(f"Visualizations saved to: {viz_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
