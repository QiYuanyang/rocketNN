import numpy as np
import json
import os
from rocketpy import Environment, SolidMotor, Rocket, Flight
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure data directory exists
os.makedirs("data/pinn_trajectories", exist_ok=True)

def extract_trajectory_data(flight, params, n_points=100):
    """
    Extract time-series trajectory data from a RocketPy Flight object.
    
    Args:
        flight: RocketPy Flight object after simulation
        params: Dictionary of input parameters
        n_points: Number of time points to sample (default: 100)
    
    Returns:
        Dictionary containing parameters and trajectory data
    """
    # Sample time points uniformly from launch to apogee
    t_array = np.linspace(0, flight.t_final, n_points)
    
    # Extract position trajectory
    x_vals = np.array([flight.x(t) for t in t_array])
    y_vals = np.array([flight.y(t) for t in t_array])
    z_vals = np.array([flight.z(t) for t in t_array])
    
    # Extract velocity trajectory
    vx_vals = np.array([flight.vx(t) for t in t_array])
    vy_vals = np.array([flight.vy(t) for t in t_array])
    vz_vals = np.array([flight.vz(t) for t in t_array])
    
    # Extract acceleration trajectory
    ax_vals = np.array([flight.ax(t) for t in t_array])
    ay_vals = np.array([flight.ay(t) for t in t_array])
    az_vals = np.array([flight.az(t) for t in t_array])
    
    # Compute physics-informed features
    wind_u = params['surface_wind_speed'] * np.sin(np.radians(params['wind_direction']))
    wind_v = params['surface_wind_speed'] * np.cos(np.radians(params['wind_direction']))
    drift_potential = params['surface_wind_speed'] * (1 + params['shear_exponent'])
    
    # Package all data
    trajectory_data = {
        # Input parameters (extended with physics features)
        'params': {
            'payload_mass': float(params['payload_mass']),
            'motor_impulse_scale': float(params['motor_impulse_scale']),
            'launch_inclination': float(params['launch_inclination']),
            'launch_heading': float(params['launch_heading']),
            'surface_wind_speed': float(params['surface_wind_speed']),
            'wind_direction': float(params['wind_direction']),
            'shear_exponent': float(params['shear_exponent']),
            # Physics-informed features
            'wind_u': float(wind_u),
            'wind_v': float(wind_v),
            'drift_potential': float(drift_potential),
        },
        
        # Time array
        'time': t_array.tolist(),
        
        # Position trajectory [x, y, z] (meters)
        'position': {
            'x': x_vals.tolist(),
            'y': y_vals.tolist(),
            'z': z_vals.tolist(),
        },
        
        # Velocity trajectory [vx, vy, vz] (m/s)
        'velocity': {
            'vx': vx_vals.tolist(),
            'vy': vy_vals.tolist(),
            'vz': vz_vals.tolist(),
        },
        
        # Acceleration trajectory [ax, ay, az] (m/s^2)
        'acceleration': {
            'ax': ax_vals.tolist(),
            'ay': ay_vals.tolist(),
            'az': az_vals.tolist(),
        },
        
        # Summary metrics
        'summary': {
            'apogee': float(flight.apogee),
            'apogee_time': float(flight.apogee_time),
            'landing_x': float(flight.x(flight.t_final)),
            'landing_y': float(flight.y(flight.t_final)),
            'flight_time': float(flight.t_final),
            'max_velocity': float(np.max(np.sqrt(vx_vals**2 + vy_vals**2 + vz_vals**2))),
        }
    }
    
    return trajectory_data


def run_pinn_simulation(num_simulations=1000, n_points=100):
    """
    Generate trajectory dataset for Soft-PINN training.
    
    Args:
        num_simulations: Number of flight simulations to run
        n_points: Number of time points per trajectory
    """
    print(f"Generating {num_simulations} trajectories for Soft-PINN training...")
    print(f"Each trajectory sampled at {n_points} time points")
    
    # Define Standard Motor Thrust Curve (Cesaroni M1670-like)
    base_thrust_source = [
        [0, 0], [0.1, 2000], [0.5, 2500], [1.0, 2200], 
        [2.0, 1800], [3.0, 1500], [4.0, 1000], [5.0, 0]
    ]
    
    # Metadata about the dataset
    metadata = {
        'num_trajectories': num_simulations,
        'points_per_trajectory': n_points,
        'parameter_ranges': {
            'payload_mass': [2.0, 6.0],
            'motor_impulse_scale': [0.95, 1.05],
            'surface_wind_speed': [0.0, 15.0],
            'wind_direction': [0.0, 360.0],
            'shear_exponent': [0.1, 0.3],
            'launch_inclination': [80.0, 90.0],
            'launch_heading': [0.0, 360.0],
        },
        'rocket_config': {
            'base_dry_mass': 14.0,
            'radius': 0.0635,
            'drag_coefficient': 0.45,
            'motor_type': 'Cesaroni M1670-like',
            'burn_time': 5.0,
        },
        'environment': {
            'latitude': 32.990254,
            'longitude': -106.974998,
            'elevation': 1400.0,
            'date': '2025-12-10',
        }
    }
    
    # Save metadata
    with open('data/pinn_trajectories/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Metadata saved to data/pinn_trajectories/metadata.json")
    
    # Generate trajectories
    successful_sims = 0
    failed_sims = 0
    
    for i in tqdm(range(num_simulations)):
        try:
            # --- 1. Sample Parameters ---
            surface_wind = np.random.uniform(0, 15)
            wind_direction = np.random.uniform(0, 360)
            shear_exponent = np.random.uniform(0.1, 0.3)
            payload_mass = np.random.uniform(2.0, 6.0)
            impulse_scale = np.random.uniform(0.95, 1.05)
            inclination = np.random.uniform(80, 90)
            heading = np.random.uniform(0, 360)
            
            params = {
                'surface_wind_speed': surface_wind,
                'wind_direction': wind_direction,
                'shear_exponent': shear_exponent,
                'payload_mass': payload_mass,
                'motor_impulse_scale': impulse_scale,
                'launch_inclination': inclination,
                'launch_heading': heading,
            }
            
            # --- 2. Setup Environment ---
            env = Environment(latitude=32.990254, longitude=-106.974998, elevation=1400)
            env.set_date((2025, 12, 10, 12))
            
            def wind_u_func(z):
                h = max(z - env.elevation, 0.1)
                speed = surface_wind * (h / 10.0) ** shear_exponent
                angle_rad = np.radians(wind_direction)
                return speed * np.sin(angle_rad)

            def wind_v_func(z):
                h = max(z - env.elevation, 0.1)
                speed = surface_wind * (h / 10.0) ** shear_exponent
                angle_rad = np.radians(wind_direction)
                return speed * np.cos(angle_rad)

            env.set_atmospheric_model(type="custom_atmosphere", wind_u=wind_u_func, wind_v=wind_v_func)
            
            # --- 3. Setup Rocket ---
            base_dry_mass = 14.0
            total_dry_mass = base_dry_mass + payload_mass
            
            # Scale thrust curve
            scaled_thrust = [[t, f * impulse_scale] for t, f in base_thrust_source]
            
            motor = SolidMotor(
                thrust_source=scaled_thrust,
                burn_time=5.0,
                grain_number=5,
                grain_separation=5/1000,
                grain_density=1815,
                grain_outer_radius=33/1000,
                grain_initial_inner_radius=15/1000,
                nozzle_radius=19/1000,
                throat_radius=11/1000,
                interpolation_method="linear",
                coordinate_system_orientation="nozzle_to_combustion_chamber",
                dry_mass=1.815,
                dry_inertia=(0.125, 0.125, 0.002),
                grain_initial_height=0.12,
                grains_center_of_mass_position=0.3,
                center_of_dry_mass_position=0.3
            )
            
            rocket = Rocket(
                radius=127/2000,
                mass=total_dry_mass,
                inertia=(6.3, 6.3, 0.034),
                power_off_drag=0.45,
                power_on_drag=0.45,
                center_of_mass_without_motor=1.0,
                coordinate_system_orientation="tail_to_nose",
            )
            
            rocket.add_motor(motor, position=-1.25)
            rocket.set_rail_buttons(upper_button_position=0.2, lower_button_position=-0.5)
            rocket.add_parachute("Main", cd_s=10.0, trigger="apogee", sampling_rate=105, lag=1.5, noise=(0, 8.3, 0.5))
            
            # --- 4. Run Simulation ---
            flight = Flight(
                rocket=rocket,
                environment=env,
                rail_length=5.2,
                inclination=inclination,
                heading=heading,
                terminate_on_apogee=False
            )
            
            # --- 5. Extract and Save Trajectory Data ---
            trajectory_data = extract_trajectory_data(flight, params, n_points=n_points)
            
            # Save to individual JSON file
            output_file = f'data/pinn_trajectories/trajectory_{i:05d}.json'
            with open(output_file, 'w') as f:
                json.dump(trajectory_data, f)
            
            successful_sims += 1
            
        except Exception as e:
            failed_sims += 1
            print(f"\nWarning: Simulation {i} failed with error: {str(e)}")
            continue
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Data generation complete!")
    print(f"Successful simulations: {successful_sims}/{num_simulations}")
    print(f"Failed simulations: {failed_sims}/{num_simulations}")
    print(f"Data saved to: data/pinn_trajectories/")
    print(f"Total data points: {successful_sims * n_points:,}")
    print(f"{'='*60}")
    
    # Create an index file for easy loading
    create_dataset_index(successful_sims)


def create_dataset_index(num_files):
    """Create an index file listing all trajectory files."""
    index = {
        'num_trajectories': num_files,
        'files': [f'trajectory_{i:05d}.json' for i in range(num_files)]
    }
    
    with open('data/pinn_trajectories/index.json', 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"Index file created: data/pinn_trajectories/index.json")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        n_sims = int(sys.argv[1])
    else:
        n_sims = 100  # Default: 100 trajectories for testing
    
    if len(sys.argv) > 2:
        n_points = int(sys.argv[2])
    else:
        n_points = 100  # Default: 100 time points per trajectory
    
    # Run data generation
    run_pinn_simulation(num_simulations=n_sims, n_points=n_points)
