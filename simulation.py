import time
import numpy as np
from metadrive.envs import MetaDriveEnv


# Standardized Metadrive benchmark presets so every model runs on identical maps
BENCHMARK_SUITES = {
    "quick_dev": {
        "description": "5 deterministic PG maps (seeds 0-4) for fast debugging",
        "start_seed": 0,
        "num_scenarios": 5,
        "map": 3,  # 3 road blocks per map
        "traffic_density": 0.1,
    },
    "fixed_course": {
        "description": "Deterministic sequence: Straight -> Curve -> Roundabout -> Ramp",
        "start_seed": 0,
        "num_scenarios": 5,
        "map": "SCrR",
        "traffic_density": 0.15,
    },
    "metadrive_official_test": {
        "description": "Official MetaDrive Generalization Test Split (seeds 0-199)",
        "start_seed": 0,
        "num_scenarios": 200,
        "map": 3,
        "traffic_density": 0.1,
    },
}


class BasePolicy:
    """common interface that all classical filters and RL policies inherit from"""
    name = "BasePolicy"

    def reset(self, env, obs, sensor_packet):
        """Called at the start of every episode"""
        pass

    def act(self, obs, sensor_packet, env):
        """returns [steering, throttle] in [-1.0, 1.0]."""
        raise NotImplementedError

    def get_estimated_pose(self):
        """optional: return [x, y, theta] if the policy performs localization."""
        return None


class SensorBus:
    """
    provides realistic noisy sensor readings alongside MetaDrive's standard obs.
    classical filters use sensor_packet; RL policies can use obs directly
    """
    def __init__(self, dt=0.02, gps_std=1.8, compass_std=0.08, speed_std=0.4, gyro_std=0.05, gps_every=5):
        self.dt = dt
        self.gps_std = gps_std
        self.compass_std = compass_std
        self.speed_std = speed_std
        self.gyro_std = gyro_std
        self.gps_every = gps_every
        self.prev_heading = 0.0

    def reset(self, vehicle):
        self.prev_heading = float(vehicle.heading_theta)

    def read(self, vehicle, obs, step):
        true_pos = np.array(vehicle.position[:2], dtype=float)
        true_theta = float(vehicle.heading_theta)
        true_speed = float(vehicle.speed)

        # High-rate (50 Hz) noisy wheel odometry + gyroscope
        d_theta = np.arctan2(
            np.sin(true_theta - self.prev_heading),
            np.cos(true_theta - self.prev_heading)
        )
        true_yaw_rate = d_theta / self.dt
        self.prev_heading = true_theta

        packet = {
            "step": step,
            "dt": self.dt,
            "odom_speed": true_speed + np.random.normal(0.0, self.speed_std),
            "gyro_yaw_rate": true_yaw_rate + np.random.normal(0.0, self.gyro_std),
            "lidar_norm": obs[-240:],  # Normalized [0, 1] 240-ray Lidar
            "gps_available": (step % self.gps_every == 0),
            "gps_xy": None,
            "compass_theta": None,
        }

        # Low-rate (10 Hz) noisy GPS + Compass
        if packet["gps_available"]:
            packet["gps_xy"] = true_pos + np.random.normal(0.0, self.gps_std, size=2)
            raw_theta = true_theta + np.random.normal(0.0, self.compass_std)
            packet["compass_theta"] = float(np.arctan2(np.sin(raw_theta), np.cos(raw_theta)))

        return packet


def run_benchmark(policy: BasePolicy, suite_name="quick_dev", render=True, max_steps=1500):
    """runs any BasePolicy across a standardized map suite and prints benchmark metrics"""
    suite = BENCHMARK_SUITES[suite_name]
    dt = 0.02

    env_config = {
        "use_render": render,
        "window_size": (1000, 800),
        "start_seed": suite["start_seed"],
        "num_scenarios": suite["num_scenarios"],
        "map": suite["map"],
        "traffic_density": suite["traffic_density"],
        "decision_repeat": 1,  # 50 Hz control/sensor loop
        "show_logo": False,
        "show_fps": True,
    }

    print(f"\n==========================================")
    print(f"Benchmark Suite : {suite_name} ({suite['description']})")
    print(f"Policy          : {policy.name}")
    print(f"==========================================\n")

    env = MetaDriveEnv(env_config)
    sensor_bus = SensorBus(dt=dt)
    results = []

    try:
        for ep in range(suite["num_scenarios"]):
            seed = suite["start_seed"] + ep
            obs, _ = env.reset(seed=seed)
            sensor_bus.reset(env.vehicle)

            init_packet = sensor_bus.read(env.vehicle, obs, step=0)
            policy.reset(env, obs, init_packet)

            pose_errors = []
            ep_reward = 0.0
            start_t = time.time()

            for step in range(max_steps):
                packet = sensor_bus.read(env.vehicle, obs, step)
                action = policy.act(obs, packet, env)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward

                # track localization error if policy estimates pose
                est_pose = policy.get_estimated_pose()
                if est_pose is not None:
                    true_xy = np.array(env.vehicle.position[:2])
                    pose_errors.append(float(np.linalg.norm(true_xy - est_pose[:2])))

                if terminated or truncated:
                    break

            elapsed = time.time() - start_t
            success = bool(info.get("arrive_dest", False))
            route_comp = float(info.get("route_completion", 0.0))
            mean_rmse = float(np.sqrt(np.mean(np.square(pose_errors)))) if pose_errors else 0.0

            results.append({
                "seed": seed,
                "success": success,
                "route_completion": route_comp,
                "reward": ep_reward,
                "rmse_m": mean_rmse,
                "steps": step + 1,
            })

            print(
                f"[Episode {ep+1}/{suite['num_scenarios']} | Seed {seed}] "
                f"Success: {success} | Route: {route_comp*100:5.1f}% | "
                f"Loc RMSE: {mean_rmse:4.2f}m | Steps: {step+1} ({elapsed:.1f}s)"
            )

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    finally:
        env.close()

    if results:
        print("\n~~~~~~ BENCHMARK SUMMARY ~~~~~~")
        print(f"Policy           : {policy.name}")
        print(f"Success Rate     : {np.mean([r['success'] for r in results])*100:.1f}%")
        print(f"Avg Route Comp   : {np.mean([r['route_completion'] for r in results])*100:.1f}%")
        print(f"Avg Reward       : {np.mean([r['reward'] for r in results]):.2f}")
        print(f"Avg Loc RMSE     : {np.mean([r['rmse_m'] for r in results]):.2f} m")


if __name__ == "__main__":
    pass