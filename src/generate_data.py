import numpy as np
import pandas as pd
from rocketpy import Environment, SolidMotor, Rocket, Flight
from tqdm import tqdm
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

def run_simulation(num_simulations=100):
    output_file = "data/flight_data.csv"
    # Start fresh to avoid appending to old data
    if os.path.exists(output_file):
        os.remove(output_file)
        
    print(f"Starting {num_simulations} simulations with Standard Calisto-like Rocket...")
    
    # Define Standard Motor Thrust Curve (Cesaroni M1670-like)
    # Time(s), Thrust(N)
    base_thrust_source = [
        [0, 0], [0.1, 2000], [0.5, 2500], [1.0, 2200], 
        [2.0, 1800], [3.0, 1500], [4.0, 1000], [5.0, 0]
    ]
    
    for i in tqdm(range(num_simulations)):
        # --- 1. Environment (Variable) ---
        # Wind Power Law parameters
        surface_wind = np.random.uniform(0, 15) # 0 to 15 m/s
        wind_direction = np.random.uniform(0, 360) 
        shear_exponent = np.random.uniform(0.1, 0.3) 
        
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
        
        # --- 2. Rocket (Fixed Design, Variable Configuration) ---
        # We use a "Calisto-like" reference rocket.
        # Geometry is FIXED.
        # Mass and Motor Performance are VARIABLE.
        
        # Variable: Payload Mass (e.g., different experiments)
        base_dry_mass = 14.0 # kg
        payload_mass = np.random.uniform(2.0, 6.0) # 2kg to 6kg payload range
        total_dry_mass = base_dry_mass + payload_mass
        
        # Variable: Motor Performance (Manufacturing variance +/- 5%)
        impulse_scale = np.random.uniform(0.95, 1.05)
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
            radius=127/2000, # Fixed 127mm
            mass=total_dry_mass, # Variable mass
            inertia=(6.3, 6.3, 0.034), # Simplified inertia
            power_off_drag=0.45, # Fixed Aerodynamics
            power_on_drag=0.45,
            center_of_mass_without_motor=1.0, # Fixed CG location
            coordinate_system_orientation="tail_to_nose",
        )
        
        rocket.add_motor(motor, position=-1.25)
        rocket.set_rail_buttons(upper_button_position=0.2, lower_button_position=-0.5)
        
        rocket.add_parachute("Main", cd_s=10.0, trigger="apogee", sampling_rate=105, lag=1.5, noise=(0, 8.3, 0.5))

        # --- 3. Mission / Launch (Variable) ---
        inclination = np.random.uniform(80, 90) 
        heading = np.random.uniform(0, 360)
        
        flight = Flight(rocket=rocket, environment=env, rail_length=5.2, inclination=inclination, heading=heading, terminate_on_apogee=False)
        
        # --- 4. Results ---
        result = {
            "surface_wind_speed": surface_wind,
            "wind_direction": wind_direction,
            "shear_exponent": shear_exponent,
            "payload_mass": payload_mass,
            "motor_impulse_scale": impulse_scale,
            "launch_inclination": inclination,
            "launch_heading": heading,
            "apogee": flight.apogee,
            "landing_x": flight.x(flight.t_final),
            "landing_y": flight.y(flight.t_final),
            "flight_time": flight.t_final
        }
        
        # Save incrementally to avoid memory issues
        df_row = pd.DataFrame([result])
        df_row.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
        
    print("Data generation complete.")
    df = pd.DataFrame(results)
    df.to_csv("data/flight_data.csv", index=False)
    print("Data generation complete.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 1000
    run_simulation(num_simulations=n)
