import cv2
import numpy as np
from scipy.spatial import cKDTree
from simulation import BasePolicy


class OccupancyGridSLAMPolicy(BasePolicy):
    """
    self-contained 2D LiDAR SLAM + Occupancy-Grid free-space controller
    -no GPS/compass used after spawn (starts at local origin [0, 0, 0]).
    -no Metadrive lane graph used for steering
    -uses ICP scan matching to correct odometry drift and builds a 2D log-odds map.
    """
    def __init__(
        self,
        grid_size=400,
        resolution=0.4, # 0.4m per cell -> 160m x 160m map
        lidar_max_dist=50.0,
        icp_iters=12,
        visualize=True,
        verbose_every=25,
    ):
        self.name = f"2D_ICP_OccupancySLAM(res={resolution}m)"
        self.grid_size = grid_size
        self.resolution = resolution
        self.origin_idx = grid_size // 2
        self.lidar_max_dist = lidar_max_dist
        self.icp_iters = icp_iters
        self.visualize = visualize
        self.verbose_every = verbose_every

        # precompute Lidar ray angles in ego frame (240 rays spanning 360 deg)
        self.num_lasers = 240
        self.ray_angles = np.linspace(0.0, 2 * np.pi, self.num_lasers, endpoint=False)

        # log-odds occupancy grid (0.0 = unknown, <0 = free, >0 = occupied)
        self.log_odds = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.l_occ = 0.85
        self.l_free = -0.35
        self.l_min = -4.0
        self.l_max = 4.0

        # SLAM pose in local map frame [x, y, theta] + transform to MetaDrive world frame for evaluation
        self.slam_pose = np.zeros(3, dtype=float)
        self.world_origin_xy = np.zeros(2, dtype=float)
        self.world_origin_theta = 0.0

        self.prev_scan_local = None
        self.traj_cells = []

    @staticmethod
    def _wrap_angle(angle):
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def world_to_grid(self, xy):
        """converts local SLAM metric coordinates (x, y) to grid indices (gx, gy)."""
        gx = np.round(xy[..., 0] / self.resolution).astype(int) + self.origin_idx
        gy = np.round(xy[..., 1] / self.resolution).astype(int) + self.origin_idx
        return gx, gy

    def reset(self, env, obs, sensor_packet):
        self.log_odds.fill(0.0)
        self.slam_pose = np.zeros(3, dtype=float)
        self.prev_scan_local = None
        self.traj_cells = []

        #save true spawn pose for simulation.py to evaluate SLAM drift against ground truth
        self.world_origin_xy = np.array(env.vehicle.position[:2], dtype=float)
        self.world_origin_theta = float(env.vehicle.heading_theta)

        # seed the initial occupancy grid from the first Lidar scan at [0, 0, 0]
        init_scan = self._lidar_to_points(obs, sensor_packet)
        self.prev_scan_local = init_scan
        self._update_occupancy_grid(self.slam_pose, init_scan)
        print("[SLAM INIT] Initialized 2D Occupancy Grid at local frame (0.0, 0.0, 0.0)")

    def get_estimated_pose(self):
        """Transforms local SLAM pose back to MetaDrive global frame for benchmark RMSE tracking."""
        c = np.cos(self.world_origin_theta)
        s = np.sin(self.world_origin_theta)
        wx = self.world_origin_xy[0] + c * self.slam_pose[0] - s * self.slam_pose[1]
        wy = self.world_origin_xy[1] + s * self.slam_pose[0] + c * self.slam_pose[1]
        wtheta = self._wrap_angle(self.world_origin_theta + self.slam_pose[2])
        return np.array([wx, wy, wtheta])

    def _lidar_to_points(self, obs, sensor_packet):        
        """
        Projects BOTH 240-ray obstacle Lidar AND road boundary rays into ego (x, y).
        """
        # obstacle lidar (last 240 dims of obs)
        lidar_dists = sensor_packet["lidar_norm"] * self.lidar_max_dist
        lidar_angles = -np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
        valid_l = (lidar_dists > 1.2) & (lidar_dists < self.lidar_max_dist - 0.5)
        pts_lidar = np.column_stack([
            lidar_dists[valid_l] * np.cos(lidar_angles[valid_l]),
            lidar_dists[valid_l] * np.sin(lidar_angles[valid_l]),
        ])

        # road-edge / lane-line detector (36 rays immediately preceding the 240 Lidar rays)
        # NOTE: if side_detector has 12 rays, slice accordingly: obs[-(240+12+36):-(240+12)]
        lane_dists = obs[-(240 + 12 + 36) : -(240 + 12)] * 20.0
        lane_angles = -np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
        valid_r = (lane_dists > 0.8) & (lane_dists < 19.5)
        pts_road = np.column_stack([
            lane_dists[valid_r] * np.cos(lane_angles[valid_r]),
            lane_dists[valid_r] * np.sin(lane_angles[valid_r]),
        ])

        if len(pts_road) > 0:
            return np.vstack([pts_lidar, pts_road])
        return pts_lidar

    def _icp_scan_match(self, src_pts, dst_pts, init_dx, init_dy, init_dtheta):
        """
        2D Point-to-Point ICP aligning current scan (`src_pts` at t) to previous scan (`dst_pts` at t-1).
        Returns refined relative ego motion [dx, dy, dtheta] in the t-1 vehicle frame.
        """
        if len(src_pts) < 15 or len(dst_pts) < 15:
            return init_dx, init_dy, init_dtheta

        dx, dy, dtheta = init_dx, init_dy, init_dtheta
        tree = cKDTree(dst_pts)

        for _ in range(self.icp_iters):
            c, s = np.cos(dtheta), np.sin(dtheta)
            R = np.array([[c, -s], [s, c]])
            transformed = (src_pts @ R.T) + np.array([dx, dy])

            dists, idx = tree.query(transformed, k=1)
            # outlier rejection: ignore points farther than 1.5m (e.g., newly visible geometry)
            inliers = dists < 1.5
            if np.sum(inliers) < 12:
                break

            A = transformed[inliers]
            B = dst_pts[idx[inliers]]

            # SVD-based Kabsch alignment step
            centroid_A = np.mean(A, axis=0)
            centroid_B = np.mean(B, axis=0)
            H = (A - centroid_A).T @ (B - centroid_B)
            U, _, Vt = np.linalg.svd(H)
            R_step = Vt.T @ U.T
            if np.linalg.det(R_step) < 0:
                Vt[1, :] *= -1
                R_step = Vt.T @ U.T

            t_step = centroid_B - R_step @ centroid_A

            # compose incremental transform
            step_angle = np.arctan2(R_step[1, 0], R_step[0, 0])
            dtheta = self._wrap_angle(dtheta + step_angle)
            new_t = R_step @ np.array([dx, dy]) + t_step
            dx, dy = float(new_t[0]), float(new_t[1])

            if np.hypot(t_step[0], t_step[1]) < 1e-4 and abs(step_angle) < 1e-4:
                break

        return dx, dy, dtheta

    def _update_occupancy_grid(self, pose, scan_pts_ego):
        """Ray-traces free space and marks occupied cells in the global log-odds grid."""
        px, py, ptheta = pose
        c, s = np.cos(ptheta), np.sin(ptheta)
        R = np.array([[c, -s], [s, c]])

        # transform ego Lidar hits into global SLAM map coordinates
        pts_world = (scan_pts_ego @ R.T) + np.array([px, py])
        start_gx, start_gy = self.world_to_grid(np.array([px, py]))
        end_gx, end_gy = self.world_to_grid(pts_world)

        # vectorized ray sampling along each beam to clear free cells
        num_samples = 45
        alphas = np.linspace(0.05, 0.95, num_samples)[:, None]  # (S, 1)
        ray_gx = np.round(start_gx + alphas * (end_gx - start_gx)).astype(int)
        ray_gy = np.round(start_gy + alphas * (end_gy - start_gy)).astype(int)

        valid_free = ((ray_gx >= 0) & (ray_gx < self.grid_size) & (ray_gy >= 0) & (ray_gy < self.grid_size))
        self.log_odds[ray_gy[valid_free], ray_gx[valid_free]] = np.clip(self.log_odds[ray_gy[valid_free], ray_gx[valid_free]] + self.l_free,self.l_min,self.l_max,)

        # mark endpoints as occupied
        valid_occ = ((end_gx >= 0) & (end_gx < self.grid_size) & (end_gy >= 0) & (end_gy < self.grid_size))
        self.log_odds[end_gy[valid_occ], end_gx[valid_occ]] = np.clip(self.log_odds[end_gy[valid_occ], end_gx[valid_occ]] + self.l_occ,self.l_min,self.l_max,)

    def _plan_from_occupancy_grid(self):
        """
        Evaluates candidate steering angles directly on the built occupancy grid
        by ray-marching forward to find the longest obstacle-free corridor.
        """
        px, py, ptheta = self.slam_pose
        candidate_angles = np.linspace(-0.75, 0.75, 19)  # radians relative to heading
        lookahead_steps = np.linspace(1.5, 18.0, 25) # meters ahead

        best_angle = 0.0
        best_score = -1e9
        best_ray_cells = []

        for rel_ang in candidate_angles:
            world_ang = ptheta + rel_ang
            ray_x = px + lookahead_steps * np.cos(world_ang)
            ray_y = py + lookahead_steps * np.sin(world_ang)
            gx, gy = self.world_to_grid(np.column_stack([ray_x, ray_y]))

            clearance_dist = lookahead_steps[-1]
            ray_cells = []
            for idx in range(len(lookahead_steps)):
                cx, cy = gx[idx], gy[idx]
                if cx < 1 or cx >= self.grid_size - 1 or cy < 1 or cy >= self.grid_size - 1:
                    clearance_dist = lookahead_steps[idx]
                    break
                # check a 3x3 inflation footprint around the ray cell for vehicle width safety
                patch = self.log_odds[cy - 1 : cy + 2, cx - 1 : cx + 2]
                if np.max(patch) > 0.4:  # occupied wall/barrier/cone
                    clearance_dist = lookahead_steps[idx]
                    break
                ray_cells.append((cx, cy))

            # score prefers maximum clear distance + slight preference for straightness
            score = clearance_dist - 2.2 * abs(rel_ang)
            if score > best_score:
                best_score = score
                best_angle = rel_ang
                best_ray_cells = ray_cells

        return float(best_angle), best_score, best_ray_cells

    def _render_slam_map(self, best_ray_cells):
        """Renders the live 2D Occupancy Grid, SLAM trajectory, and chosen corridor."""
        # convert log-odds to grayscale image (127 = unknown, 255 = free, 0 = occupied)
        prob = 1.0 / (1.0 + np.exp(-self.log_odds))
        img_gray = ((1.0 - prob) * 255).astype(np.uint8)
        canvas = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        # draw SLAM trajectory (green)
        gx, gy = self.world_to_grid(self.slam_pose[:2])
        self.traj_cells.append((gx, gy))
        for i in range(1, len(self.traj_cells)):
            cv2.line(canvas, self.traj_cells[i - 1], self.traj_cells[i], (0, 220, 0), 1)

        # draw chosen free-space planning ray (cyan)
        for cx, cy in best_ray_cells:
            canvas[cy, cx] = (255, 255, 0)

        # draw ego pose dot (red)
        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            cv2.circle(canvas, (gx, gy), 3, (0, 0, 255), -1)

        # needs to be flipped to match map orientation and upscale for visibility
        canvas = cv2.flip(canvas, 0)
        canvas = cv2.resize(canvas, (600, 600), interpolation=cv2.INTER_NEAREST)
        cv2.putText(
            canvas,
            f"SLAM Pose: ({self.slam_pose[0]:.1f}m, {self.slam_pose[1]:.1f}m)",
            (10, 25), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 180, 0), 
            2,
        )
        
        cv2.imshow("2D Occupancy Grid SLAM (Live Map)", canvas)
        cv2.waitKey(1)

    def act(self, obs, sensor_packet, env):
        step = sensor_packet["step"]
        dt = sensor_packet["dt"]

        # 1) odometry initial guess in ego frame
        odom_dx = sensor_packet["odom_speed"] * dt
        odom_dtheta = sensor_packet["gyro_yaw_rate"] * dt

        # 2) scan-to-scan ICP correction using 240-ray Lidar
        curr_scan = self._lidar_to_points(obs, sensor_packet)
        dx_ego, dy_ego, dtheta = self._icp_scan_match(curr_scan, self.prev_scan_local, odom_dx, 0.0, odom_dtheta)
        self.prev_scan_local = curr_scan

        # 3) integrate corrected relative motion into global SLAM pose
        theta = self.slam_pose[2]
        c, s = np.cos(theta), np.sin(theta)
        self.slam_pose[0] += c * dx_ego - s * dy_ego
        self.slam_pose[1] += s * dx_ego + c * dy_ego
        self.slam_pose[2] = self._wrap_angle(theta + dtheta)

        # 4) update Occupancy Grid
        self._update_occupancy_grid(self.slam_pose, curr_scan)

        # 5) plan steering & speed strictly from our own built Occupancy Grid
        best_angle, clear_score, best_ray_cells = self._plan_from_occupancy_grid()
        raw_steer = float(np.clip(1.35 * best_angle, -1.0, 1.0))
        self.prev_steer = getattr(self, "prev_steer", 0.0)
        steering = 0.65 * self.prev_steer + 0.35 * raw_steer 
        self.prev_steer = steering

        # slow down in tight curves / near obstacles on the grid
        target_speed = 8.5 if clear_score > 11.0 else 4.5
        if clear_score < 4.0:
            throttle = -0.8
        else:
            throttle = float(np.clip(0.35 * (target_speed - sensor_packet["odom_speed"]), -0.5, 0.55))

        if self.visualize:
            self._render_slam_map(best_ray_cells)

        if step % self.verbose_every == 0:
            true_xy = np.array(env.vehicle.position[:2])
            est_world_xy = self.get_estimated_pose()[:2]
            drift_err = float(np.linalg.norm(true_xy - est_world_xy))
            occ_cells = int(np.sum(self.log_odds > 0.4))
            free_cells = int(np.sum(self.log_odds < -0.4))
            print(
                f"  [SLAM Step {step:04d}] LocalPose: ({self.slam_pose[0]:5.1f}, {self.slam_pose[1]:5.1f}) | "
                f"Odom-ICP Drift: {drift_err:4.2f}m | Map Cells (Occ/Free): {occ_cells}/{free_cells}"
            )

        return [steering, throttle]