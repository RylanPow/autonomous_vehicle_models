import time
import numpy as np
from metadrive.envs import MetaDriveEnv


BENCHMARK_SUITES = {
    "quick_dev": {
        "description": "5 deterministic PG maps (seeds 0-4) for fast debugging",
        "start_seed": 0,
        "num_scenarios": 5,
        "map": 3,
        "traffic_density": 0.1,
    },
    "fixed_course": {
        "description": "Deterministic sequence: Straight -> Curve -> Roundabout -> Ramp",
        "start_seed": 0,
        "num_scenarios": 5,
        "map": "SCrR",
        "traffic_density": 0.15,
    },
    "slam_static_loop": {
        "description": "Zero traffic + static obstacles & roundabout for SLAM loop closure",
        "start_seed": 0,
        "num_scenarios": 3,
        "map": "CrOC",
        "traffic_density": 0.0,
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
    """Common interface for classical filters, SLAM, and RL policies."""
    name = "BasePolicy"

    def reset(self, env, obs, sensor_packet):
        pass

    def act(self, obs, sensor_packet, env):
        raise NotImplementedError

    def get_estimated_pose(self):
        """Optional: return [x, y, theta] if the policy performs localization."""
        return None


class SensorBus:
    """Provides realistic noisy sensor readings alongside MetaDrive's standard `obs`."""
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
            "lidar_norm": obs[-240:],
            "gps_available": (step % self.gps_every == 0),
            "gps_xy": None,
            "compass_theta": None,
        }

        if packet["gps_available"]:
            packet["gps_xy"] = true_pos + np.random.normal(0.0, self.gps_std, size=2)
            raw_theta = true_theta + np.random.normal(0.0, self.compass_std)
            packet["compass_theta"] = float(np.arctan2(np.sin(raw_theta), np.cos(raw_theta)))

        return packet


class EpisodeMetricsTracker:
    """Tracks AV safety, lane-keeping, comfort, localization, and latency metrics."""
    def __init__(self, dt):
        self.dt = dt
        self.reset()

    def reset(self):
        self.speeds = []
        self.accels = []
        self.jerks = []
        self.cte_errors = [] # cross-track error (meters from lane center)
        self.steering_rates = []
        self.loc_errors = []
        self.inference_ms = []
        self.hard_brakes = 0
        self.near_misses = 0  # forward TTC < 1.5s
        self.prev_speed = 0.0
        self.prev_accel = 0.0
        self.prev_steer = 0.0

    def record_step(self, vehicle, action, lidar_norm, est_pose, infer_time_s):
        speed = float(vehicle.speed)
        steer = float(action[0])

        # 1) longitudinal kinematics and comfort (acceleration and jerk)
        accel = (speed - self.prev_speed) / self.dt
        jerk = (accel - self.prev_accel) / self.dt
        steer_rate = abs(steer - self.prev_steer) / self.dt

        self.speeds.append(speed)
        self.accels.append(accel)
        self.jerks.append(jerk)
        self.steering_rates.append(steer_rate)
        self.inference_ms.append(infer_time_s * 1000.0)

        if accel < -3.5:
            self.hard_brakes += 1

        self.prev_speed = speed
        self.prev_accel = accel
        self.prev_steer = steer

        # 2) continuous lane-keeping Cross-Track Error (CTE)
        try:
            lane = vehicle.navigation.current_ref_lanes[0]
            _, lat_offset = lane.local_coordinates(vehicle.position[:2])
            self.cte_errors.append(abs(float(lat_offset)))
        except Exception:
            pass

        # 3) near-miss Time-To-Collision (TTC) check in front cone
        front_dist = float(np.min(np.concatenate([lidar_norm[:8], lidar_norm[-8:]]) * 50.0))
        if speed > 1.0 and (front_dist / speed) < 1.5:
            self.near_misses += 1

        # 4) localization error (if policy outputs an estimated pose)
        if est_pose is not None:
            true_xy = np.array(vehicle.position[:2])
            self.loc_errors.append(float(np.linalg.norm(true_xy - est_pose[:2])))

    def summarize(self, info, termination_reason, steps):
        impact_speed = self.speeds[-1] if "crash" in termination_reason else 0.0
        return {
            "termination": termination_reason,
            "success": termination_reason == "arrive_dest",
            "route_completion": float(info.get("route_completion", 0.0)),
            "steps": steps,
            "impact_speed_ms": impact_speed,
            "near_miss_steps": self.near_misses,
            "cte_rms_m": float(np.sqrt(np.mean(np.square(self.cte_errors)))) if self.cte_errors else 0.0,
            "cte_max_m": float(np.max(self.cte_errors)) if self.cte_errors else 0.0,
            "jerk_rms_ms3": float(np.sqrt(np.mean(np.square(self.jerks)))) if self.jerks else 0.0,
            "steer_rate_rms": float(np.sqrt(np.mean(np.square(self.steering_rates)))) if self.steering_rates else 0.0,
            "hard_brakes": self.hard_brakes,
            "loc_rmse_m": float(np.sqrt(np.mean(np.square(self.loc_errors)))) if self.loc_errors else 0.0,
            "infer_ms": float(np.mean(self.inference_ms)) if self.inference_ms else 0.0,
        }


def classify_termination(info, terminated, truncated, custom_reason):
    if custom_reason:
        return custom_reason
    if info.get("arrive_dest", False):
        return "arrive_dest"
    if info.get("crash_vehicle", False):
        return "crash_vehicle"
    if info.get("crash_object", False):
        return "crash_object"
    if info.get("crash_building", False) or info.get("crash", False):
        return "crash_other"
    if info.get("out_of_road", False):
        return "out_of_road"
    if truncated:
        return "max_steps_timeout"
    return "unknown"


def run_benchmark(
    policy: BasePolicy,
    suite_name="quick_dev",
    render=True,
    max_steps=1500,
    terminate_on_collision=True,
    stagnation_limit_steps=200,
):
    suite = BENCHMARK_SUITES[suite_name]
    dt = 0.02

    env_config = {
        "use_render": render,
        "window_size": (1000, 800),
        "start_seed": suite["start_seed"],
        "num_scenarios": suite["num_scenarios"],
        "map": suite["map"],
        "traffic_density": suite["traffic_density"],
        "decision_repeat": 1,
        "show_logo": False,
        "show_fps": True,
    }

    print(f"\n============================================================")
    print(f"AV Validation Suite : {suite_name} | Policy: {policy.name}")
    print(f"============================================================\n")

    env = MetaDriveEnv(env_config)
    sensor_bus = SensorBus(dt=dt)
    tracker = EpisodeMetricsTracker(dt=dt)
    all_summaries = []

    try:
        for ep in range(suite["num_scenarios"]):
            seed = suite["start_seed"] + ep
            obs, _ = env.reset(seed=seed)
            sensor_bus.reset(env.vehicle)
            tracker.reset()

            init_packet = sensor_bus.read(env.vehicle, obs, step=0)
            policy.reset(env, obs, init_packet)

            idle_steps = 0
            custom_term = None

            for step in range(max_steps):
                packet = sensor_bus.read(env.vehicle, obs, step)

                t0 = time.perf_counter()
                action = policy.act(obs, packet, env)
                infer_s = time.perf_counter() - t0

                obs, _, terminated, truncated, info = env.step(action)
                est_pose = policy.get_estimated_pose()
                tracker.record_step(env.vehicle, action, packet["lidar_norm"], est_pose, infer_s)

                # custom AV termination checks: stagnation & filter divergence
                if env.vehicle.speed < 0.2:
                    idle_steps += 1
                    if idle_steps >= stagnation_limit_steps:
                        custom_term = "stagnation_timeout"
                        break
                else:
                    idle_steps = 0

                if est_pose is not None:
                    loc_err = np.linalg.norm(np.array(env.vehicle.position[:2]) - est_pose[:2])
                    if loc_err > 12.0:
                        custom_term = "localization_diverged"
                        break

                if terminated or truncated:
                    if not terminate_on_collision and info.get("crash", False) and not info.get("out_of_road", False):
                        continue
                    break

            term_reason = classify_termination(info, terminated, truncated, custom_term)
            summary = tracker.summarize(info, term_reason, step + 1)
            all_summaries.append(summary)

            print(
                f"[Seed {seed:03d}] End: {summary['termination']:18s} | "
                f"Route: {summary['route_completion']*100:5.1f}% | "
                f"CTE RMS: {summary['cte_rms_m']:4.2f}m | "
                f"Jerk RMS: {summary['jerk_rms_ms3']:5.2f} | "
                f"Loc RMSE: {summary['loc_rmse_m']:4.2f}m | "
                f"Infer: {summary['infer_ms']:4.2f}ms"
            )

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    finally:
        env.close()

    if all_summaries:
        n = len(all_summaries)
        term_counts = {}
        for s in all_summaries:
            term_counts[s["termination"]] = term_counts.get(s["termination"], 0) + 1

        print("\n==================== AV VALIDATION REPORT ====================")
        print(f"Policy                  : {policy.name} ({n} episodes)")
        print(f"Success Rate            : {np.mean([s['success'] for s in all_summaries])*100:5.1f}%")
        print(f"Avg Route Completion    : {np.mean([s['route_completion'] for s in all_summaries])*100:5.1f}%")
        print(f"Termination Breakdown   : {term_counts}")
        print(f"Avg Impact Speed        : {np.mean([s['impact_speed_ms'] for s in all_summaries]):5.2f} m/s")
        print(f"Lane CTE (RMS / Max)    : {np.mean([s['cte_rms_m'] for s in all_summaries]):4.2f}m / {np.max([s['cte_max_m'] for s in all_summaries]):4.2f}m")
        print(f"Comfort Jerk RMS        : {np.mean([s['jerk_rms_ms3'] for s in all_summaries]):5.2f} m/s^3")
        print(f"Steering Rate RMS       : {np.mean([s['steer_rate_rms'] for s in all_summaries]):5.2f} rad/s")
        print(f"Localization RMSE       : {np.mean([s['loc_rmse_m'] for s in all_summaries]):4.2f} m")
        print(f"Policy Latency          : {np.mean([s['infer_ms'] for s in all_summaries]):5.2f} ms/step")
        print("==============================================================\n")


if __name__ == "__main__":
    from particle_filter_policy import ParticleFilterPolicy

    pf_policy = ParticleFilterPolicy(num_particles=300, verbose_every=50)
    run_benchmark(pf_policy, suite_name="quick_dev", render=True)