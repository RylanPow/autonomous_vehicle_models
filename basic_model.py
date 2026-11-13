from metadrive.envs import MetaDriveEnv

def main():
    env = MetaDriveEnv(config={
        "use_render": True, # show the 3D window
        "manual_control": False,
        "num_scenarios": 1, # fix one map for testing
        "start_seed": 0,
    })
    obs, info = env.reset()
    for step in range(500):
        # 1. read sensor data from obs, info, env
        # 2. run custom algorithm here (Kalman, policy, etc.)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()

if __name__ == "__main__":
    main()