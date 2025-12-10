import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# 1. Define the Neural Network (Deeper and Wider)
class RocketSurrogateModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RocketSurrogateModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128), # Added Batch Norm for stability
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
        
    def forward(self, x):
        return self.network(x)

# 2. Custom Dataset Class
class RocketDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(data_path="data/flight_data.csv", epochs=100):
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Define Inputs (Features) and Outputs (Targets)
    feature_cols = [
        "surface_wind_speed", "wind_direction", "shear_exponent",
        "payload_mass", "motor_impulse_scale", 
        "launch_inclination", "launch_heading"
    ]
    target_cols = ["apogee", "landing_x", "landing_y"]
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize Data (Crucial for NN performance)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Create DataLoaders
    train_dataset = RocketDataset(X_train_scaled, y_train_scaled)
    test_dataset = RocketDataset(X_test_scaled, y_test_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize Model
    model = RocketSurrogateModel(input_size=len(feature_cols), output_size=len(target_cols))
    criterion = nn.MSELoss()
    # Reduced Learning Rate for deeper network
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # Training Loop
    print("Starting training (Deeper Model)...")
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        test_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
    # Evaluation & Visualization
    print("Training complete. Evaluating...")
    model.eval()
    with torch.no_grad():
        # Predict on full test set
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_pred_scaled = model(X_test_tensor).numpy()
        
        # Inverse transform to get real units
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
    # Calculate Metrics
    mse_apogee = np.mean((y_test[:, 0] - y_pred[:, 0])**2)
    rmse_apogee = np.sqrt(mse_apogee)
    
    dist_error = np.sqrt((y_test[:, 1] - y_pred[:, 1])**2 + (y_test[:, 2] - y_pred[:, 2])**2)
    mean_dist_error = np.mean(dist_error)
    
    print(f"\n--- Results ---")
    print(f"Apogee RMSE: {rmse_apogee:.2f} meters")
    print(f"Average Landing Location Error: {mean_dist_error:.2f} meters")
    
    # Plotting
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (Scaled)')
    plt.title('Training Convergence')
    plt.legend()
    
    # Plot 2: Landing Prediction vs Actual
    plt.subplot(1, 2, 2)
    plt.scatter(y_test[:, 1], y_test[:, 2], c='blue', label='Actual (RocketPy)', alpha=0.5, s=10)
    plt.scatter(y_pred[:, 1], y_pred[:, 2], c='red', label='Predicted (NN)', alpha=0.5, s=10)
    plt.xlabel('Landing X (m)')
    plt.ylabel('Landing Y (m)')
    plt.title('Landing Location: Actual vs Predicted')
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig("results/training_results.png")
    print("Results plot saved to results/training_results.png")
    
    # Save Model
    torch.save(model.state_dict(), "results/rocket_surrogate_model.pth")
    print("Model saved to results/rocket_surrogate_model.pth")

if __name__ == "__main__":
    train_model()
