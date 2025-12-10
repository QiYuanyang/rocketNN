# Rocket Flight Prediction with Neural Networks

This project implements a **Physics-Informed Neural Network (PINN)** approach to predict the flight trajectory of high-power rockets. It uses **RocketPy** to generate high-fidelity training data and **PyTorch** to train a surrogate model that can predict Apogee and Landing Location in real-time.

## 🚀 Project Overview

- **Simulator:** RocketPy (6-DOF Physics Engine)
- **Model:** PyTorch Neural Network (Split Architecture)
- **Goal:** Predict flight outcomes based on variable wind and rocket configurations.

## 📂 Structure

- `src/generate_data.py`: Generates synthetic flight data using RocketPy.
- `src/train_model.py`: Trains a baseline Neural Network.
- `src/train_split_model.py`: Trains the advanced Split-Model (ApogeeNet + LandingNet) with physics features.
- `data/`: Contains generated CSV datasets (ignored by git).
- `results/`: Contains model checkpoints and evaluation plots (ignored by git).

## 🛠️ Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install torch scikit-learn
   ```

2. **Generate Data:**
   ```bash
   # Generate 10,000 simulations
   python src/generate_data.py 10000
   ```

3. **Train Model:**
   ```bash
   python src/train_split_model.py
   ```

## 📊 Results

The model achieves:
- **Apogee RMSE:** ~33 meters
- **Landing Error:** ~670 meters (due to high wind drift uncertainty)

## 📝 License

MIT License
