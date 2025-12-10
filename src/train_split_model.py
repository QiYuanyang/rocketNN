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

# --- 1. Physics-Informed Feature Engineering ---
def preprocess_data(df):
    """
    Adds physics-informed features to the dataset.
    """
    # Convert Wind Polar -> Cartesian (U, V)
    # U = East (X), V = North (Y)
    # Angle is in degrees, need radians
    wind_rad = np.radians(df['wind_direction'])
    
    df['wind_u'] = df['surface_wind_speed'] * np.sin(wind_rad)
    df['wind_v'] = df['surface_wind_speed'] * np.cos(wind_rad)
    
    # Interaction Feature: "Drift Potential"
    # High Shear * High Wind = More Drift
    df['drift_potential'] = df['surface_wind_speed'] * (1 + df['shear_exponent'])
    
    return df

# --- 2. Define Specialized Models ---

class ApogeeNet(nn.Module):
    """Specialized for predicting Altitude (Scalar)"""
    def __init__(self, input_size):
        super(ApogeeNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output: Apogee
        )
        
    def forward(self, x):
        return self.network(x)

class LandingNet(nn.Module):
    """Specialized for predicting Location (X, Y)"""
    def __init__(self, input_size):
        super(LandingNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 2) # Output: X, Y
        )
        
    def forward(self, x):
        return self.network(x)

# --- 3. Dataset Class ---
class RocketDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_split_models(data_path="data/flight_data.csv", epochs=200):
    print("Loading and Preprocessing Data...")
    df = pd.read_csv(data_path)
    df = preprocess_data(df)
    
    # --- Define Features for Each Model ---
    
    # Apogee depends mostly on Rocket Config + Launch Angle
    # Wind affects it slightly (weathercocking), so we include wind.
    apogee_features = [
        "payload_mass", "motor_impulse_scale", 
        "launch_inclination", "launch_heading",
        "surface_wind_speed", "shear_exponent" # Wind magnitude matters, direction less so for altitude
    ]
    
    # Landing depends on EVERYTHING + Wind Vectors
    landing_features = [
        "payload_mass", "motor_impulse_scale", 
        "launch_inclination", "launch_heading",
        "wind_u", "wind_v", "shear_exponent", "drift_potential"
    ]
    
    # Targets
    y_apogee_raw = df[['apogee']].values
    y_landing_raw = df[['landing_x', 'landing_y']].values
    
    # Prepare Data Matrices
    X_apogee = df[apogee_features].values
    X_landing = df[landing_features].values
    
    # Split
    # We use the same indices for both so we can evaluate them together later
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Scalers
    scaler_apogee_X = StandardScaler().fit(X_apogee[train_idx])
    scaler_landing_X = StandardScaler().fit(X_landing[train_idx])
    
    scaler_apogee_y = StandardScaler().fit(y_apogee_raw[train_idx])
    scaler_landing_y = StandardScaler().fit(y_landing_raw[train_idx])
    
    # --- Train Apogee Model ---
    print("\n=== Training Apogee Model ===")
    train_loader = DataLoader(RocketDataset(
        scaler_apogee_X.transform(X_apogee[train_idx]), 
        scaler_apogee_y.transform(y_apogee_raw[train_idx])
    ), batch_size=64, shuffle=True)
    
    model_apogee = ApogeeNet(input_size=len(apogee_features))
    opt_apogee = optim.Adam(model_apogee.parameters(), lr=0.001)
    crit = nn.MSELoss()
    
    for epoch in range(epochs):
        model_apogee.train()
        for X_b, y_b in train_loader:
            opt_apogee.zero_grad()
            loss = crit(model_apogee(X_b), y_b)
            loss.backward()
            opt_apogee.step()
            
    # --- Train Landing Model ---
    print("\n=== Training Landing Model ===")
    train_loader = DataLoader(RocketDataset(
        scaler_landing_X.transform(X_landing[train_idx]), 
        scaler_landing_y.transform(y_landing_raw[train_idx])
    ), batch_size=64, shuffle=True)
    
    model_landing = LandingNet(input_size=len(landing_features))
    opt_landing = optim.Adam(model_landing.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model_landing.train()
        for X_b, y_b in train_loader:
            opt_landing.zero_grad()
            loss = crit(model_landing(X_b), y_b)
            loss.backward()
            opt_landing.step()

    # --- Evaluation ---
    print("\n=== Final Evaluation ===")
    model_apogee.eval()
    model_landing.eval()
    
    with torch.no_grad():
        # Predict Apogee
        X_test_apogee = torch.FloatTensor(scaler_apogee_X.transform(X_apogee[test_idx]))
        pred_apogee_scaled = model_apogee(X_test_apogee).numpy()
        pred_apogee = scaler_apogee_y.inverse_transform(pred_apogee_scaled)
        
        # Predict Landing
        X_test_landing = torch.FloatTensor(scaler_landing_X.transform(X_landing[test_idx]))
        pred_landing_scaled = model_landing(X_test_landing).numpy()
        pred_landing = scaler_landing_y.inverse_transform(pred_landing_scaled)
        
        # Ground Truth
        true_apogee = y_apogee_raw[test_idx]
        true_landing = y_landing_raw[test_idx]
        
        # Metrics
        rmse_apogee = np.sqrt(np.mean((true_apogee - pred_apogee)**2))
        dist_error = np.sqrt(np.sum((true_landing - pred_landing)**2, axis=1))
        mean_dist_error = np.mean(dist_error)
        
        print(f"Apogee RMSE: {rmse_apogee:.2f} m")
        print(f"Landing Error: {mean_dist_error:.2f} m")
        
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.scatter(true_landing[:, 0], true_landing[:, 1], c='blue', alpha=0.3, label='Actual')
        plt.scatter(pred_landing[:, 0], pred_landing[:, 1], c='red', alpha=0.3, label='Predicted')
        plt.title(f"Landing Prediction (Split Model)\nMean Error: {mean_dist_error:.1f}m")
        plt.xlabel("East (m)")
        plt.ylabel("North (m)")
        plt.legend()
        plt.axis('equal')
        plt.savefig("results/split_model_results.png")
        print("Plot saved to results/split_model_results.png")

if __name__ == "__main__":
    train_split_models()
