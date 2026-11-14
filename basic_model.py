import time
import random
import numpy as np
from metadrive.envs import MetaDriveEnv


# ==========================================
# TUNABLE ENVIRONMENT CONFIGURATION
# ==========================================
class Config:
    # map settings
    MAP_SEED = 42         
    NUM_SCENARIOS = 1 # only load 1 map
    
    # object and traffic settings
    TRAFFIC_DENSITY = 1.0 # 0.0 = empty road, 1.0 = heavy traffic (max is ~0.4)
    
    # --- Render Settings ---
    RENDER_3D = True         # Set to False for faster, headless execution
    WINDOW_SIZE = (1000, 800)

def basic_policy(step_count):
    """
    example policy
    -always applies 50% throttle.
    -every 10 steps it randomly either steers hard or brakes
    -action space in metadrive is [steering, throttle] (both -1.0 to 1.0)
    """
    throttle = 0.5
    steering = 0.0
    
    # every 10 steps, do something erratic so you can see the physics
    if step_count % 10 == 0 and step_count > 0:
        action_choice = random.choice(["brake", "steer_left", "steer_right"])
        
        if action_choice == "brake":
            throttle = -1.0  # Hard brake
        elif action_choice == "steer_left":
            steering = 0.5   # Steer left
        elif action_choice == "steer_right":
            steering = -0.5  # Steer right
            
    return [steering, throttle]

def main():
    env_config = {
        "use_render": Config.RENDER_3D,
        "window_size": Config.WINDOW_SIZE,
        "num_scenarios": Config.NUM_SCENARIOS,
        "start_seed": Config.MAP_SEED,
        
        # traffic and objects
        "traffic_density": Config.TRAFFIC_DENSITY,

        # ui options
        "show_logo": False,
        "show_fps": True,
    }

    print("Initializing MetaDrive...")
    env = MetaDriveEnv(env_config)
    
    # reset the environment
    obs, info = env.reset()
    print("Environment Reset. Starting simulation.")
    print("Action Space:", env.action_space)
    
    step = 0
    try:
        # simulation loop
        while True:
            # get the action from  policy
            action = basic_policy(step)
            
            # step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # grab ground truth data
            vehicle_pos = env.vehicle.position
            vehicle_speed = env.vehicle.speed
            
            # print status every 10 steps
            if step % 10 == 0:
                print(f"Step: {step:04d} | Speed: {vehicle_speed:.2f} m/s | Pos: ({vehicle_pos[0]:.1f}, {vehicle_pos[1]:.1f}) | Action: {action}")
            
            # if the car crashes or gets stuck, reset the simulation
            if terminated or truncated:
                print(f"\n--- Episode ended at step {step}. Resetting... ---\n")
                obs, info = env.reset()
                step = 0
            else:
                step += 1
            

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        env.close()
        print("Environment closed.")


if __name__ == "__main__":
    main()