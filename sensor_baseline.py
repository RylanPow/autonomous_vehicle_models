import cv2
import numpy as np
from metadrive.envs import MetaDriveEnv
from metadrive.obs.state_obs import LidarStateObservation
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.component.sensors.semantic_camera import SemanticCamera


class Config:
    MAP_SEED = 42
    NUM_SCENARIOS = 1
    TRAFFIC_DENSITY = 0.25

    RENDER_3D = True
    WINDOW_SIZE = (1000, 800)
    CAM_RES = (320, 180)  # (width, height) for attached vehicle cameras

    # lidar safety controller settings
    NUM_LASERS = 240
    LIDAR_MAX_DIST = 50.0  # meters
    STOP_THRESHOLD_M = 12.0  # brake if obstacle is closer than 12m in front


def lidar_aeb_policy(lidar_distances_m):
    """
    autonomous emergency braking (AEB) demo policy using Lidar rays
    -240 Lidar rays sweep 360 degrees around the car
    -index 0 points straight ahead; rays wrap around [0... 239].
    """
    # grab 30 degree forward cone (first and last 10 rays)
    front_cone = np.concatenate([lidar_distances_m[:10], lidar_distances_m[-10:]])
    min_front_dist = float(np.min(front_cone))

    steering = 0
    if min_front_dist < Config.STOP_THRESHOLD_M:
        throttle = -1.0  # full brake
        status = "BRAKING (OBSTACLE)"
    else:
        throttle = 0.4   # cruise forward
        status = "CRUISING"

    return [steering, throttle], min_front_dist, status


def main():
    env_config = {
        "use_render": Config.RENDER_3D,
        "window_size": Config.WINDOW_SIZE,
        "num_scenarios": Config.NUM_SCENARIOS,
        "start_seed": Config.MAP_SEED,
        "traffic_density": Config.TRAFFIC_DENSITY,

        # keep standard 1D state + lidar vector as 'obs', while still rendering cameras
        "image_observation": True,
        "agent_observation": LidarStateObservation,

        # attach virtual sensors to the ego vehicle
        "sensors": {
            "rgb_cam": (RGBCamera, *Config.CAM_RES),
            "depth_cam": (DepthCamera, *Config.CAM_RES),
            "semantic_cam": (SemanticCamera, *Config.CAM_RES),
        },

        # vehicle & lidar hardware config + 3D visual overlays
        "vehicle_config": {
            "lidar": {
                "num_lasers": Config.NUM_LASERS,
                "distance": Config.LIDAR_MAX_DIST,
                "num_others": 4,  # also detect 4 nearest vehicles' states
            },
            "show_lidar": True, # draws green/red Lidar rays in 3D window
            "show_side_detector": True, # draws side proximity beams
            "show_lane_line_detector": True,
        },

        "show_logo": False,
        "show_fps": True,
    }

    print("Initializing MetaDrive with Multi-Sensor Suite...")
    env = MetaDriveEnv(env_config)
    obs, info = env.reset()

    # grab direct handles to the attached sensors
    rgb_sensor = env.engine.get_sensor("rgb_cam")
    depth_sensor = env.engine.get_sensor("depth_cam")
    semantic_sensor = env.engine.get_sensor("semantic_cam")
    lidar_sensor = env.engine.get_sensor("lidar")

    print("Available Sensors:", list(env.engine.sensors.keys()))
    print("Observation Shape (Ego + Navi + Lidar):", obs.shape)

    step = 0
    try:
        while True:
            # extract normalized Lidar readings [0.0, 1.0] from the tail of 'obs'
            # and convert to real-world meters
            lidar_norm = obs[-Config.NUM_LASERS:]
            lidar_meters = lidar_norm * Config.LIDAR_MAX_DIST

            # compute policy action based on front lidar cone
            action, min_front_dist, status = lidar_aeb_policy(lidar_meters)

            # step
            obs, reward, terminated, truncated, info = env.step(action)

            # query raw lidar point cloud and detected objects directly (may be useful for SLAM)
            cloud_points, detected_objs = lidar_sensor.perceive(
                env.vehicle,
                physics_world=env.engine.physics_world.dynamic_world,
                num_lasers=Config.NUM_LASERS,
                distance=Config.LIDAR_MAX_DIST,
            )

            # query camera buffers (returns float32 [0, 1] or uint8 HxWxC arrays)
            rgb_img = rgb_sensor.perceive(to_float=False) # HxWx3 (BGR/RGB)
            depth_img = depth_sensor.perceive(to_float=True) # HxWx1 float [0, 1]
            sem_img = semantic_sensor.perceive(to_float=False) # HxWx3 class colors

            # convert depth (1 channel) to 3 channel uint8 so can stack all 3 feeds
            depth_vis = (np.repeat(depth_img, 3, axis=-1) * 255).astype(np.uint8)
            combined_view = np.hstack([rgb_img, depth_vis, sem_img])

            cv2.imshow("Ego Sensors: RGB | Depth | Semantic", combined_view)
            # required for openCV window to refresh; press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # telemetry print every 10 steps
            if step % 10 == 0:
                print(
                    f"Step {step:04d} | Speed: {env.vehicle.speed:5.2f} m/s | "
                    f"Front Lidar Min: {min_front_dist:5.2f}m | "
                    f"Detected Objs: {len(detected_objs)} | Status: {status}"
                )

            if terminated or truncated:
                print(f"\n--- Episode ended at step {step} ({info.get('crash', 'done')}). Resetting... ---\n")
                obs, info = env.reset()
                step = 0
            else:
                step += 1

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        cv2.destroyAllWindows()
        env.close()
        print("Environment and sensor windows closed.")


if __name__ == "__main__":
    main()