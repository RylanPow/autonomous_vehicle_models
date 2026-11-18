import numpy as np
from simulation import BasePolicy


class ParticleFilterPolicy(BasePolicy):
    """
    2D Monte Carlo Localization (Particle Filter) + pure pursuit & Lidar AEB Policy
    state per particle: [x, y, theta]
    """
    def __init__(self, num_particles=300, verbose_every=25):
        self.name = f"ParticleFilter(N={num_particles})"
        self.N = num_particles
        self.verbose_every = verbose_every

        # particle cloud: shape (N, 3) -> [x, y, theta] and weights shape (N, )
        self.particles = np.zeros((self.N, 3), dtype=float)
        self.weights = np.ones(self.N, dtype=float) / self.N
        self.estimated_pose = np.zeros(3, dtype=float)

        # motion & sensor noise hyperparameters
        self.motion_noise_xy = 0.12     # process scatter per step (meters)
        self.motion_noise_theta = 0.02  # heading scatter per step (rad)
        self.gps_sigma = 1.8            # measurement likelihood std (meters)
        self.compass_sigma = 0.12       # heading likelihood std (rad)

    @staticmethod
    def _wrap_angle(angles):
        return np.arctan2(np.sin(angles), np.cos(angles))

    def reset(self, env, obs, sensor_packet):
        """Initialize particle cloud around the first GPS + Compass reading."""
        init_xy = sensor_packet["gps_xy"]
        init_theta = sensor_packet["compass_theta"]

        self.particles[:, 0] = init_xy[0] + np.random.normal(0.0, 1.0, self.N)
        self.particles[:, 1] = init_xy[1] + np.random.normal(0.0, 1.0, self.N)
        self.particles[:, 2] = self._wrap_angle(
            init_theta + np.random.normal(0.0, 0.1, self.N)
        )
        self.weights.fill(1.0 / self.N)
        self.estimated_pose = np.array([init_xy[0], init_xy[1], init_theta])

        print(f"[PF INIT] Spawned {self.N} particles around ({init_xy[0]:.1f}, {init_xy[1]:.1f})")

    def predict(self, odom_speed, gyro_yaw_rate, dt):
        """STEP 1: Propagate all N particles forward through noisy kinematic motion."""
        # add per-particle control noise so particles explore plausible trajectories
        noisy_v = odom_speed + np.random.normal(0.0, 0.35, self.N)
        noisy_w = gyro_yaw_rate + np.random.normal(0.0, 0.05, self.N)

        theta = self.particles[:, 2]
        self.particles[:, 0] += noisy_v * np.cos(theta) * dt + np.random.normal(0.0, self.motion_noise_xy, self.N)
        self.particles[:, 1] += noisy_v * np.sin(theta) * dt + np.random.normal(0.0, self.motion_noise_xy, self.N)
        self.particles[:, 2] = self._wrap_angle(
            theta + noisy_w * dt + np.random.normal(0.0, self.motion_noise_theta, self.N)
        )

    def update(self, gps_xy, compass_theta):
        """STEP 2: score particles against incoming GPS & compass measurements"""
        dx = self.particles[:, 0] - gps_xy[0]
        dy = self.particles[:, 1] - gps_xy[1]
        dist_sq = dx**2 + dy**2

        dtheta = self._wrap_angle(self.particles[:, 2] - compass_theta)

        # log-likelihood for numerical stability before normalizing
        log_likelihood = (
            -0.5 * (dist_sq / (self.gps_sigma**2))
            -0.5 * ((dtheta**2) / (self.compass_sigma**2))
        )
        log_likelihood -= np.max(log_likelihood)  # Shift for numerical stability
        likelihood = np.exp(log_likelihood)

        self.weights *= likelihood
        weight_sum = np.sum(self.weights)
        if weight_sum > 1e-12:
            self.weights /= weight_sum
        else:
            # recovery fallback if all particles drifted too far
            self.weights.fill(1.0 / self.N)

    def resample_if_needed(self):
        """STEP 3: Systematic Resampling when Effective Sample Size (N_eff) drops."""
        n_eff = 1.0 / np.sum(self.weights**2)
        resampled = False

        if n_eff < self.N / 2.0:
            # Systematic low-variance resampling
            positions = (np.arange(self.N) + np.random.uniform()) / self.N
            cumulative_sum = np.cumsum(self.weights)
            cumulative_sum[-1] = 1.0  # Guard against float rounding
            indexes = np.searchsorted(cumulative_sum, positions)

            self.particles = self.particles[indexes]
            # Roughening: add tiny jitter so identical cloned particles diversify
            self.particles[:, 0] += np.random.normal(0.0, 0.08, self.N)
            self.particles[:, 1] += np.random.normal(0.0, 0.08, self.N)
            self.particles[:, 2] = self._wrap_angle(
                self.particles[:, 2] + np.random.normal(0.0, 0.015, self.N)
            )
            self.weights.fill(1.0 / self.N)
            resampled = True

        return n_eff, resampled

    def estimate_state(self):
        """STEP 4: Weighted mean of particle cloud (using circular mean for heading)."""
        est_x = float(np.sum(self.particles[:, 0] * self.weights))
        est_y = float(np.sum(self.particles[:, 1] * self.weights))

        # circular weighted mean for angle
        sin_sum = float(np.sum(np.sin(self.particles[:, 2]) * self.weights))
        cos_sum = float(np.sum(np.cos(self.particles[:, 2]) * self.weights))
        est_theta = float(np.arctan2(sin_sum, cos_sum))

        self.estimated_pose = np.array([est_x, est_y, est_theta])
        return self.estimated_pose

    def get_estimated_pose(self):
        return self.estimated_pose

    def act(self, obs, sensor_packet, env):
        step = sensor_packet["step"]

        # predict (every tick at 50hz)
        self.predict(
            sensor_packet["odom_speed"],
            sensor_packet["gyro_yaw_rate"],
            sensor_packet["dt"]
        )

        # update and resample when 10hz GPS/compass arrives
        n_eff, resampled = None, False
        if sensor_packet["gps_available"]:
            self.update(sensor_packet["gps_xy"], sensor_packet["compass_theta"])
            n_eff, resampled = self.resample_if_needed()

        # 3. compute weighted pose estimate
        est_x, est_y, est_theta = self.estimate_state()

        # diagnostics for monitoring PF
        if step % self.verbose_every == 0:
            spread_std = float(np.mean(np.std(self.particles[:, :2], axis=0)))
            true_xy = env.vehicle.position[:2]
            err = float(np.linalg.norm(true_xy - self.estimated_pose[:2]))
            neff_str = f"{n_eff:5.1f}" if n_eff is not None else "  N/A"
            print(
                f"  [PF Step {step:04d}] Est: ({est_x:6.1f}, {est_y:6.1f}) | "
                f"Err: {err:4.2f}m | Cloud Spread: {spread_std:4.2f}m | "
                f"N_eff: {neff_str} | Resampled: {resampled}"
            )

        # closed loop pure pursuit + Lidar AEB driven ONLY by estimated_pose
        lane = env.vehicle.navigation.current_ref_lanes[0]
        long_pos, _ = lane.local_coordinates([est_x, est_y])
        target_pt = lane.position(long_pos + 12.0, 0.0)

        target_angle = np.arctan2(target_pt[1] - est_y, target_pt[0] - est_x)
        heading_err = float(self._wrap_angle(target_angle - est_theta))
        steering = float(np.clip(1.8 * heading_err, -1.0, 1.0))

        lidar_m = sensor_packet["lidar_norm"] * 50.0
        min_front = float(np.min(np.concatenate([lidar_m[:10], lidar_m[-10:]])))

        if min_front < 12.0:
            throttle = -1.0
        else:
            speed_err = 10.0 - sensor_packet["odom_speed"]
            throttle = float(np.clip(0.3 * speed_err, -0.5, 0.6))

        return [steering, throttle]