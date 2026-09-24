import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
import pybullet_planning as pp
from pybullet_planning import (rrt_connect, get_distance_fn, get_sample_fn, get_extend_fn, get_collision_fn, smooth_path)
from pybullet_planning import cartesian_motion_planning


class KinematicChainMotionPlanner:
    def __init__(self, robot, target_positions=None, target_joint_configs=None):
        """
        Initialize the motion planner with target positions and backend.
        """
        self.robot = robot
        self.target_positions = target_positions
        self.target_joint_configs = target_joint_configs

    def manipulability_gradient(self, joint_positions, delta=1e-4):
        """ Numerically compute manipulability gradient with respect to joint positions """
        joint_positions = list(joint_positions)
        w0 = self.robot.safe_manipulability(joint_positions)
        grad = np.zeros_like(joint_positions)
        for i in range(len(joint_positions)):
            q_delta = np.array(joint_positions, dtype=float)
            q_delta[i] += delta
            w1 = self.robot.safe_manipulability(q_delta)
            grad[i] = (w1 - w0) / delta
        return grad
    
    def joint_limit_avoidance_gradient(self, joint_positions, margin=0.5):
        """
        Compute a repulsive gradient pushing joints away from their limits.
        The closer to a limit, the stronger the gradient.
        """
        grad = np.zeros_like(joint_positions)
        for i, q in enumerate(joint_positions):
            q_min = self.robot.lower_limits[i]
            q_max = self.robot.upper_limits[i]
            q_range = q_max - q_min
            q_center = (q_max + q_min) / 2.0
            buffer = margin * q_range

            # Repulsive gradient (e.g., quadratic or inverse barrier function)
            if q < q_min + buffer:
                grad[i] = (q_min + buffer - q) / (buffer**2)
            elif q > q_max - buffer:
                grad[i] = (q_max - buffer - q) / (buffer**2)
            else:
                grad[i] = 0
        return grad

    def shortest_angular_distance(self, start_configuration, end_configuration):
        """
        Calculate the shortest angular distance between start and end joint configurations.

        Parameters:
        - start_configuration: list or numpy array of initial joint angles
        - end_configuration: list or numpy array of target joint angles

        Returns:
        - adjusted_end_configuration: numpy array of end joint angles modified to take the shortest angular distance to the start configuration
        """
        start_configuration = np.array(start_configuration, dtype=float)
        end_configuration = np.array(end_configuration, dtype=float)

        # Wrap only the difference, never the start itself: wrapping the start into [-pi, pi]
        # teleports any joint sitting outside that range (e.g. wrist 1 at 5pi/6 drifting past pi)
        # by 2pi, which shows up as a full-turn jump in the path.
        shortest_difference = (end_configuration - start_configuration + np.pi) % (2 * np.pi) - np.pi

        return start_configuration + shortest_difference

    def interpolate_joint_trajectory(self, start_config, end_config, num_steps, collision_objects=None):
        """
        Interpolates a joint joint trajectory from start_config to end_config

        Parameters:
        - start_config: numpy array of shape (n,), start joint positions
        - end_config: numpy array of shape (n,), end joint positions
        - num_steps: int, number of interpolation steps
        - collision_objects: optional list of body IDs to also check each waypoint against.
          Defaults to None, meaning only self-collision is checked (the original behavior) -
          pass this to check environment collision in the same pass instead of a separate loop.

        Returns:
        - interpolated_configs: numpy array of shape (num_steps, n), interpolated joint positions
        """
        
        start_config = np.array(start_config)
        end_config = np.array(end_config)

        # Minimize angular rotation of the last two joints
        end_config[4:] = self.shortest_angular_distance(start_config[4:], end_config[4:])

        # Create an array for the interpolated configurations
        interpolated_configs = np.zeros((num_steps, len(start_config)))

        # Loop over each joint to interpolate using minimal angular changes
        for j in range(len(start_config)):
            # Interpolate linearly between the start and adjusted end angles
            for i in range(num_steps):
                interpolated_value = np.linspace(start_config[j], end_config[j], num_steps)[i]
                
                # Ensure the joint value stays within [-2pi, 2pi] after interpolation
                interpolated_value = np.clip(interpolated_value, self.robot.lower_limits[j], self.robot.upper_limits[j])
                
                # Store the interpolated value in the array
                interpolated_configs[i, j] = interpolated_value

        # Check for collisions in the interpolated path (self-collision always; environment
        # collision too, in the same pass, if collision_objects is given)
        collision_in_path = any(
            self.robot.check_self_collision(config) or
            (collision_objects is not None and self.robot.collision_check(self.robot.robotId, collision_objects))
            for config in interpolated_configs
        )

        return interpolated_configs, collision_in_path

    def resolved_rate_control(self, target_pose, alpha=0.75, max_steps=10000, tol=0.05,
                            manipulability_gain=0.1, damping_lambda=0.15, beta=0.9, max_joint_vel=1.0,
                            stall_patience=10, stall_vel_threshold=0.1, plot_manipulability=False):
        """
        Resolved-rate motion control with manipulability maximization and smoothing.
        Args:
            target_pose (tuple): (target_position, target_orientation)
            alpha (float): Scaling factor for joint velocity
            max_steps (int): Maximum number of iterations
            tol (float): Tolerance for convergence. Used for position and orientation errors
            manipulability_gain (float): Gain for manipulability nullspace biasing
            damping_lambda (float): Damping factor for the damped least squares pseudoinverse
            beta (float): Velocity smoothing factor (low-pass filter)
            max_joint_vel (float): Maximum joint velocity (rad/s)
            stall_patience (int): Number of steps to wait before considering motion stalled
            stall_vel_threshold (float): Velocity threshold for stalling
            plot_manipulability (bool): Whether to plot manipulability over time
        Returns:
            tuple: (final joint configuration as list, integrated manipulability over time)
               if target pose is reached within tolerance, else (None, integrated manipulability).
        """
        dq_prev = np.zeros(len(self.robot.get_joint_positions()))  # For filtering
        manipulability_history = []
        joint_change_history = []
        stall_counter = 0

        for step in range(max_steps):
            q = np.array(self.robot.get_joint_positions())
            J = self.robot.get_jacobian(q)

            # Store manipulability
            manipulability = self.robot.safe_manipulability(q)
            manipulability_history.append(manipulability)

            # ✅ Calculate current pose error
            current_pos, current_ori = self.robot.get_link_state(self.robot.end_effector_index)
            target_pos, target_ori = target_pose
            pose_within_tol, pos_err_axis, pos_err_norm, ori_err_axis, ori_err_angle = self.robot.check_pose_within_tolerance(current_pos, current_ori, target_pos, target_ori, tol)

            vel_ee = np.hstack((pos_err_axis, np.array(ori_err_axis) * ori_err_angle))

            # ✅ Damped least squares pseudoinverse
            # Make sure the Jacobian is square
            JT = J.T
            JJt = J @ JT
            # Add damping term to help prevent instabilities (especially near singularities)
            lambda_I = damping_lambda**2 * np.eye(J.shape[0])
            J_pinv = JT @ np.linalg.inv(JJt + lambda_I)

            dq_main = J_pinv @ vel_ee # Initial joint velocity command update

            #TODO: Do I need to do joint limit avoidance here?
            # ✅ Nullspace biasing
            N = np.eye(len(q)) - J_pinv @ J
            grad_w = self.manipulability_gradient(q)
            dq_manip_bias = manipulability_gain * N @ grad_w

            # ✅ Sum the bias terms with the primary command
            dq = dq_main + dq_manip_bias
            # ✅ Nullspace biasing
            # N = np.eye(len(q)) - J_pinv @ J
            # grad_w = self.manipulability_gradient(q)
            # grad_joint_limits = self.joint_limit_avoidance_gradient(q)

            # dq_manip_bias = manipulability_gain * N @ grad_w
            # dq_limit_bias = manipulability_gain * N @ grad_joint_limits 

            # # ✅ Sum the bias terms with the primary command
            # dq = dq_main + dq_manip_bias + dq_limit_bias

            # ✅ Clip joint velocities
            dq = np.clip(dq, -max_joint_vel, max_joint_vel)

            # ✅ Exponential smoothing on dq (low pass filter)
            dq_filtered = beta * dq + (1 - beta) * dq_prev
            dq_prev = dq_filtered

            # Calculate change in joint angles for this step (delta_q = alpha * dq_filtered)
            delta_q = alpha * dq_filtered
            joint_change = np.linalg.norm(delta_q)
            joint_change_history.append(joint_change)

            # ✅ Joint update
            q_new = q + alpha * dq_filtered
            self.robot.reset_joint_positions(q_new.tolist(), step_sim=True)

            # Text
            self.robot.con.addUserDebugText(f"Manipulability: {manipulability:.4f}", [0.4, 0, 0], [0.5, 0.0, 0.8], 1.5, 0.1)

            # print(f"[INFO] Step {step}: Position Error: {np.linalg.norm(pos_err)}, Orientation Error: {np.abs(ori_err_angle)}")

            # ✅ Check for convergence
            if pose_within_tol:
                # print(f"[INFO] Converged in {step} steps.")
                break

            # ✅ Check for motion stalling
            if np.linalg.norm(dq_filtered) < stall_vel_threshold:
                stall_counter += 1
                if stall_counter >= stall_patience:
                    # print(f"[WARN] Motion stalled for {stall_patience} consecutive steps. Terminating early.")
                    break
            else:
                stall_counter = 0  # Reset if motion resumes
        # else:
        #     print("[WARN] Max steps reached without full convergence.")
        
        # Plot after control loop
        if plot_manipulability:
            plt.figure(figsize=(8, 4))
            plt.plot(manipulability_history, label='Manipulability Index', color='dodgerblue')
            plt.xlabel("Timestep")
            plt.ylabel("Manipulability")
            plt.title("Manipulability Over Time")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

        # Compute integrated scores using numerical integration (e.g., via the trapezoidal rule)
        integrated_manipulability = np.trapz(manipulability_history) * alpha
        integrated_joint_change = np.trapz(joint_change_history) * alpha

        return q_new.tolist() if pose_within_tol else None, integrated_manipulability, integrated_joint_change, (pos_err_norm, ori_err_angle)
        
    
    def interpolate_joint_trajectory2(self, start_config, end_config, num_steps):
        """
        Interpolates a joint trajectory from start_config to end_config ensuring but ensuring the base joint actuates first

        Parameters:
        - start_config: numpy array of shape (n,), start joint positions
        - end_config: numpy array of shape (n,), end joint positions
        - num_steps: int, number of interpolation steps

        Returns:
        - interpolated_configs: numpy array of shape (num_steps, n), interpolated joint positions
        """
        
        start_config = np.array(start_config)
        end_config = np.array(end_config)

        # Minimize angular rotation of the last two joints
        end_config[4:] = self.shortest_angular_distance(start_config[4:], end_config[4:])

        # Create an array for the interpolated configurations
        interpolated_configs = np.zeros((num_steps, len(start_config)))

        # Calculate the number of steps for the first joint (5%)
        first_joint_steps = int(num_steps * 0.3)
        
        # Interpolation for the first joint
        for i in range(first_joint_steps):
            interpolated_value = np.linspace(start_config[0], end_config[0], first_joint_steps)[i]
            interpolated_value = np.clip(interpolated_value, self.robot.lower_limits[0], self.robot.upper_limits[0])
            interpolated_configs[i, 0] = interpolated_value

        # Set remaining positions for the other joints
        for j in range(1, len(start_config)):
            # Interpolate linearly between the start and adjusted end angles for other joints
            for i in range(num_steps - first_joint_steps):
                interpolated_value = np.linspace(start_config[j], end_config[j], num_steps)[i]
                interpolated_value = np.clip(interpolated_value, self.robot.lower_limits[j], self.robot.upper_limits[j])
                interpolated_configs[i + first_joint_steps, j] = interpolated_value

        # Ensure the first positions of other joints remain the same
        for i in range(first_joint_steps):
            interpolated_configs[i, 1:] = start_config[1:]

        interpolated_configs[first_joint_steps:, 0] = interpolated_configs[first_joint_steps - 1, 0]

        # Check for collisions in the interpolated path
        collision_in_path = any(self.robot.check_self_collision(config) for config in interpolated_configs)

        return interpolated_configs, collision_in_path

    def _two_stage_waypoints(self, start_position, end_position, num_steps):
        """ Yields (i, position) pairs for a two-stage world X/Z-then-Y sweep between two
        end-effector positions: stage 1 moves X/Z with Y held at `start_position`'s value,
        stage 2 moves Y with X/Z held at `end_position`'s value. i runs from 1 to num_steps - 2
        inclusive; the caller owns waypoints 0 (start) and num_steps - 1 (forced to the end).
        """
        num_steps_stage1 = max(2, num_steps // 2)
        num_steps_stage2 = num_steps - num_steps_stage1

        x_wp = np.linspace(start_position[0], end_position[0], num_steps_stage1)
        z_wp = np.linspace(start_position[2], end_position[2], num_steps_stage1)
        for i in range(1, num_steps_stage1):
            yield i, np.array([x_wp[i], start_position[1], z_wp[i]])

        y_wp = np.linspace(start_position[1], end_position[1], num_steps_stage2)
        for j in range(1, num_steps_stage2):
            yield num_steps_stage1 - 1 + j, np.array([end_position[0], y_wp[j], end_position[2]])

    def two_stage_cartesian_path(self, start_config, end_config, num_steps=100,
                                  collision_objects=None, pos_tol=1e-3, ik_max_iter=200):
        """ Plans a two-stage Cartesian path between two joint configurations.

        Stage 1 sweeps the end-effector through the world X/Z plane (depth and height)
        while holding world-Y fixed at its starting value. Stage 2 then sweeps world-Y
        (reaching out to the side) while holding X/Z fixed at the values reached at the
        end of stage 1. End-effector orientation is slerped continuously across both stages.

        Args:
            start_config (array-like): starting joint configuration
            end_config (array-like): target joint configuration
            num_steps (int, optional): total number of waypoints across both stages. Defaults to 100.
            collision_objects (list, optional): body IDs to check the path against.
                Defaults to the robot's configured collision_objects.
            pos_tol (float, optional): IK position tolerance. Defaults to 1e-3.
            ik_max_iter (int, optional): max IK iterations per waypoint. Defaults to 200.

        Returns:
            joint_path (np.ndarray): (num_steps, n) array of joint configurations.
            collision_in_path (bool): True if any waypoint is in self- or environment-collision.
        """
        start_config = np.array(start_config, dtype=float)
        end_config = np.array(end_config, dtype=float)

        start_position, start_orientation = self.robot.get_ee_pose(start_config)
        end_position, end_orientation = self.robot.get_ee_pose(end_config)
        self.robot.reset_joint_positions(start_config)

        slerp = Slerp([0, 1], R.from_quat([start_orientation, end_orientation]))

        joint_path = np.zeros((num_steps, len(start_config)))
        joint_path[0] = start_config
        prev_config = start_config

        for i, position in self._two_stage_waypoints(start_position, end_position, num_steps):
            orientation = slerp(i / (num_steps - 1)).as_quat()
            joint_config = self.robot.inverse_kinematics((position, orientation), pos_tol=pos_tol,
                                                           rest_config=list(prev_config), max_iter=ik_max_iter)
            joint_config = self.shortest_angular_distance(prev_config, joint_config)
            joint_path[i] = joint_config
            prev_config = joint_config

        # Force the final waypoint to match the requested end configuration exactly
        joint_path[-1] = self.shortest_angular_distance(joint_path[-2], end_config)

        collision_in_path = any(self.robot.in_collision(config, collision_objects) for config in joint_path)

        return joint_path, collision_in_path

    def two_stage_cartesian_path_avoid_collisions(self, start_config, end_config, num_steps=100,
                                                   collision_objects=None, pos_tol=1e-3, ik_max_iter=200,
                                                   max_retries=10, perturb_scale=0.1, stop_on_collision=True):
        """ Same staging as `two_stage_cartesian_path` (world X/Z, then world Y), but with
        collision avoidance built into the planner: each waypoint's IK solve is retried,
        perturbing the rest configuration, until it finds a self- and environment-collision-free
        solution or `max_retries` is exhausted.

        Args:
            start_config (array-like): starting joint configuration
            end_config (array-like): target joint configuration
            num_steps (int, optional): total number of waypoints across both stages. Defaults to 100.
            collision_objects (list, optional): body IDs to check each waypoint against.
                Defaults to the robot's configured collision_objects.
            pos_tol (float, optional): IK position tolerance. Defaults to 1e-3.
            ik_max_iter (int, optional): max IK iterations per waypoint. Defaults to 200.
            max_retries (int, optional): max collision-avoidance retries per waypoint. Defaults to 10.
            perturb_scale (float, optional): magnitude (rad) of the random rest-config perturbation
                applied between retries. Defaults to 0.1.
            stop_on_collision (bool, optional): if True, planning halts at the first waypoint that
                can't be resolved collision-free within `max_retries`, and the returned path is
                truncated up to (and not including) that waypoint. If False, planning continues
                using the best (still colliding) solution found for that waypoint. Defaults to True.

        Returns:
            joint_path (np.ndarray): array of joint configurations. Shorter than num_steps if
                stop_on_collision truncated the path.
            collision_in_path (bool): True if any returned waypoint is in self- or
                environment-collision.
        """
        start_config = np.array(start_config, dtype=float)
        end_config = np.array(end_config, dtype=float)

        start_position, start_orientation = self.robot.get_ee_pose(start_config)
        end_position, end_orientation = self.robot.get_ee_pose(end_config)
        self.robot.reset_joint_positions(start_config)

        # Resolved to a concrete list (not None) since it's passed straight through to
        # inverse_kinematics, where collision_objects=None means "skip the env check".
        collision_objects = collision_objects if collision_objects is not None else self.robot.collision_objects

        slerp = Slerp([0, 1], R.from_quat([start_orientation, end_orientation]))

        joint_path = np.zeros((num_steps, len(start_config)))
        joint_path[0] = start_config
        prev_config = start_config
        collision_in_path = False

        for i, position in self._two_stage_waypoints(start_position, end_position, num_steps):
            orientation = slerp(i / (num_steps - 1)).as_quat()
            joint_config, collision_free = self.robot.inverse_kinematics(
                (position, orientation), pos_tol=pos_tol, rest_config=list(prev_config),
                max_iter=ik_max_iter, num_resample=max_retries,
                collision_objects=collision_objects, perturb_scale=perturb_scale, return_status=True)
            joint_config = self.shortest_angular_distance(prev_config, joint_config)
            joint_path[i] = joint_config
            prev_config = joint_config
            if not collision_free:
                collision_in_path = True
                if stop_on_collision:
                    return joint_path[:i], True

        # Force the final waypoint to match the requested end configuration exactly
        joint_path[-1] = self.shortest_angular_distance(joint_path[-2], end_config)
        if self.robot.in_collision(end_config, collision_objects):
            collision_in_path = True

        return joint_path, collision_in_path

    def _approach_waypoints(self, start_position, start_orientation, goal_position, goal_orientation,
                            approach_axis, approach_dist, lin_res, ang_res):
        """ Builds dense (position, Rotation) waypoints for a retract -> traverse -> approach path.

        With `a` the goal's approach axis in the world frame and s(p) = a . p the progress along
        it, the path runs through the clearance plane s = s_c, s_c = min(s(start), s(goal) - approach_dist):
          A. retract along -a from the start until s = s_c (empty if the start is already behind it)
          B. traverse within that plane to the approach line through the goal
          C. approach along +a to the goal, covering at least `approach_dist`
        Orientation is slerped from start to goal across A+B and held at the goal orientation
        through C, so the final approach is a pure straight-line motion.
        """
        goal_rotation = R.from_quat(goal_orientation)
        a = goal_rotation.apply(np.asarray(approach_axis, dtype=float))
        a /= np.linalg.norm(a)

        s_start = a @ start_position
        s_goal = a @ goal_position
        s_clear = min(s_start, s_goal - approach_dist)
        retract_position = start_position - (s_start - s_clear) * a
        pre_approach_position = goal_position - (s_goal - s_clear) * a

        # Stages A+B as one polyline, parameterized by arc length so orientation progresses evenly
        corners = np.array([start_position, retract_position, pre_approach_position])
        seg_lengths = np.linalg.norm(np.diff(corners, axis=0), axis=1)
        cum_lengths = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        ab_length = cum_lengths[-1]

        slerp = Slerp([0, 1], R.concatenate([R.from_quat(start_orientation), goal_rotation]))
        ab_angle = (goal_rotation * R.from_quat(start_orientation).inv()).magnitude()
        n_ab = max(1, int(np.ceil(ab_length / lin_res)), int(np.ceil(ab_angle / ang_res)))

        waypoints = []
        for t in np.linspace(0, 1, n_ab + 1)[1:]:
            if ab_length > 0:
                d = t * ab_length
                k = min(np.searchsorted(cum_lengths, d, side='right') - 1, len(seg_lengths) - 1)
                frac = 0.0 if seg_lengths[k] == 0 else (d - cum_lengths[k]) / seg_lengths[k]
                position = corners[k] + frac * (corners[k + 1] - corners[k])
            else:
                position = start_position  # pure in-place reorientation
            waypoints.append((position, slerp(t)))

        # Stage C: straight approach at the fixed goal orientation
        c_length = s_goal - s_clear
        n_c = max(1, int(np.ceil(c_length / lin_res)))
        for t in np.linspace(0, 1, n_c + 1)[1:]:
            waypoints.append((pre_approach_position + t * (goal_position - pre_approach_position), goal_rotation))

        return waypoints

    def _local_ik(self, q, target_position, target_rotation, pos_tol, ori_tol, max_iter, damping):
        """ Damped-least-squares IK seeded at `q`. Converges to the solution nearest `q` and so
        never hops IK branches, unlike PyBullet's global IK. Returns (q, converged, manipulability). """
        lower, upper = np.array(self.robot.lower_limits), np.array(self.robot.upper_limits)
        for _ in range(max_iter + 1):
            position, orientation = self.robot.forward_kinematics(q)
            pos_err = target_position - position
            ori_err = (target_rotation * R.from_quat(orientation).inv()).as_rotvec()
            J = self.robot.get_jacobian(q)
            JJt = J @ J.T
            if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(ori_err) < ori_tol:
                return q, True, np.sqrt(max(np.linalg.det(JJt), 0.0))
            dq = J.T @ np.linalg.solve(JJt + damping**2 * np.eye(6), np.concatenate((pos_err, ori_err)))
            q = np.clip(q + dq, lower, upper)
        return q, False, 0.0

    def approach_cartesian_path(self, start_config, goal_pose, num_steps=100, approach_dist=0.15,
                                approach_axis=(0, 0, 1), collision_objects=None, lin_res=0.005,
                                ang_res=np.deg2rad(2), max_joint_step=0.1, min_manipulability=0.003,
                                pos_tol=1e-3, ori_tol=np.deg2rad(0.5), ik_max_iter=20, damping=0.01):
        """ Plans a retract -> traverse -> approach Cartesian path (see `_approach_waypoints`)
        and tracks it with local IK, so the joint path is continuous by construction.

        Waypoints are tracked densely (every `lin_res` m / `ang_res` rad), each solve seeded
        from the previous one. Rather than ever jumping to another IK branch, the path is
        rejected if any waypoint fails to converge, moves a joint more than `max_joint_step`,
        drops below `min_manipulability` (i.e. passes near a singularity, where branch flips
        happen), or is in collision. The goal is given as a pose, not a configuration: the goal
        configuration is wherever the continuous path ends up, i.e. `joint_path[-1]`.

        Args:
            start_config (array-like): starting joint configuration
            goal_pose (tuple): (position, quaternion xyzw) of the end-effector at the goal
            num_steps (int, optional): number of waypoints in the returned path. Defaults to 100.
            approach_dist (float, optional): minimum length (m) of the final straight approach. Defaults to 0.15.
            approach_axis (array-like, optional): approach direction in the end-effector frame. Defaults to +z.
            collision_objects (list, optional): body IDs to check against. Defaults to the robot's collision_objects.
            lin_res (float, optional): tracking resolution (m). Defaults to 0.005.
            ang_res (float, optional): tracking resolution (rad). Defaults to 2 deg.
            max_joint_step (float, optional): max joint change (rad) between dense waypoints. Defaults to 0.1.
            min_manipulability (float, optional): Yoshikawa manipulability floor along the path. For the UR5e,
                0.003 is roughly 6 deg from the elbow or wrist singularity. Defaults to 0.003.
            pos_tol, ori_tol (float, optional): per-waypoint IK tolerances (m, rad).
            ik_max_iter (int, optional): max DLS iterations per waypoint. Defaults to 20.
            damping (float, optional): DLS damping. Defaults to 0.01.

        Returns:
            joint_path (np.ndarray or None): (num_steps, n) joint path, or None on failure.
            info (dict): 'status' ('ok', 'ik_failed', 'joint_jump', 'low_manipulability' or
                'collision'), 'min_manipulability' along the tracked portion, and 'failed_at'
                (dense waypoint index, or None).
        """
        collision_objects = collision_objects if collision_objects is not None else self.robot.collision_objects
        q = np.array(start_config, dtype=float)
        start_position, start_orientation = self.robot.forward_kinematics(q)
        goal_position, goal_orientation = np.asarray(goal_pose[0], dtype=float), np.asarray(goal_pose[1], dtype=float)

        waypoints = self._approach_waypoints(start_position, start_orientation, goal_position, goal_orientation,
                                             approach_axis, approach_dist, lin_res, ang_res)

        dense_path = [q]
        min_manip = self.robot.calculate_manipulability(q)
        info = {'status': 'ok', 'min_manipulability': min_manip, 'failed_at': None}

        for k, (position, rotation) in enumerate(waypoints):
            q_next, converged, manip = self._local_ik(q, position, rotation, pos_tol, ori_tol, ik_max_iter, damping)
            min_manip = min(min_manip, manip)
            info['min_manipulability'] = min_manip
            if not converged:
                status = 'ik_failed'
            elif np.max(np.abs(q_next - q)) > max_joint_step:
                status = 'joint_jump'
            elif manip < min_manipulability:
                status = 'low_manipulability'
            elif self.robot.in_collision(q_next, collision_objects):
                status = 'collision'
            else:
                q = q_next
                dense_path.append(q)
                continue
            info.update(status=status, failed_at=k)
            return None, info

        return self.sample_path_to_length(dense_path, num_steps), info

    def sample_path_to_length(self, path, desired_length):
        """ Takes a joint trajectory path of any length and interpolates to a desired array length

        Args:
            path (float list): joint trajectory
            desired_length (int): desired length of trajectory (number of rows)

        Returns:
            float list: joint trajectory of desired length
        """
        path = np.array(path)
        current_path_len = path.shape[0] # Number of rows
        num_joints = path.shape[1] # Numer of columns

        # Generate new indices for interpolation
        new_indices = np.linspace(0, current_path_len - 1, desired_length)

        # Interpolate each column separately
        return np.array([np.interp(new_indices, np.arange(current_path_len), path[:, i]) for i in range(num_joints)]).T

    def make_strict_collision_fn(self, obstacles):
        # in_collision resets the robot to q itself (via check_self_collision), so there's no
        # need to separately drive it there with set_joint_configuration first.
        def fn(q):
            return self.robot.in_collision(q, obstacles)
        return fn

    def rrt_path(self, start_joint_config, end_joint_config, collision_objects=None, steps=None, rrt_iter=500,
                 joint_weights=None, smooth=True, smooth_iterations=100):
        """ Plans a joint-space path with RRT-Connect.

        Args:
            start_joint_config (array-like): starting joint configuration
            end_joint_config (array-like): target joint configuration
            collision_objects (list, optional): body IDs to check the path against
            steps (int, optional): resample the final path to this many waypoints
            rrt_iter (int, optional): max RRT-Connect iterations. Defaults to 500.
            joint_weights (array-like, optional): per-joint weights for the distance metric used
                to grow/connect the RRT trees. Higher weight makes RRT-Connect more reluctant to
                move that joint, so weighting the proximal joints (shoulder/elbow) higher than the
                wrist joints biases the search away from large shoulder/elbow swings. Defaults to
                None (uniform weights, i.e. plain joint-space Euclidean distance).
            smooth (bool, optional): whether to run a post-hoc shortcutting/smoothing pass on the
                raw RRT-Connect path to remove the detours/backtracking RRT's random sampling tends
                to leave behind. Defaults to True.
            smooth_iterations (int, optional): max shortcutting iterations for the smoothing pass.
                Defaults to 100.

        Returns:
            path (list or None): joint-space path, or None if the start/end configuration is in
                collision or RRT-Connect found no path.
        """
        extend_fn = get_extend_fn(self.robot.robotId, self.robot.controllable_joint_idx)
        # collision_fn = get_collision_fn(self.robot.robotId, self.robot.controllable_joint_idx, collision_objects)
        collision_fn = self.make_strict_collision_fn(collision_objects)
        distance_fn = get_distance_fn(self.robot.robotId, self.robot.controllable_joint_idx, weights=joint_weights)
        sample_fn = get_sample_fn(self.robot.robotId, self.robot.controllable_joint_idx)

        # Step 1: Early Exit - If Start is Already Close to Any Goal - Compute Euclidean distance (L2 norm)
        if np.linalg.norm(np.array(start_joint_config) - np.array(end_joint_config)) < 0.1:
            # print("Start configuration is already close to the goal. No need for RRT.")
            return [start_joint_config, end_joint_config]

        # Step 2: Early Collision Check
        if collision_fn(start_joint_config):
            # print("Start configuration is in collision. Skipping RRT.")
            return None
        elif collision_fn(end_joint_config):
            # print("End configuration is in collision. Skipping RRT.")
            return None

        path = rrt_connect(
            start_joint_config, end_joint_config,
            extend_fn=extend_fn,
            collision_fn=collision_fn,
            distance_fn=distance_fn,
            sample_fn=sample_fn,
            max_iterations=rrt_iter
        )

        # Step 3: Shortcut/smooth the raw path to remove RRT's characteristic detours
        if path and smooth:
            path = smooth_path(path, extend_fn=extend_fn, collision_fn=collision_fn,
                                distance_fn=distance_fn, max_smooth_iterations=smooth_iterations)

        # Ensure the path has exactly `steps` joint configurations
        if path and steps: 
            path = self.sample_path_to_length(path, steps)
        
        return path

    def plan_cartesian_motion_path(self, waypoint_poses, max_iterations=200, custom_limits={}, get_sub_conf=False, **kwargs):
        """
        Plans a Cartesian motion path along a series of end-effector waypoints 
        using pybullet_planning.cartesian_motion_planning.plan_cartesian_motion.
        
        Parameters
        ----------
        waypoint_poses : list
            A list of end-effector poses. Each pose should be a tuple (position, orientation),
            where position is a 3-element list (or array) and orientation is a 4-element quaternion.
        max_iterations : int, optional
            Maximum iterations per waypoint (default is 200).
        custom_limits : dict, optional
            Custom joint limits dictionary to be passed to the planner (default {}).
        get_sub_conf : bool, optional
            If True, returns the sub-kinematics chain configuration (default False).
        **kwargs : dict
            Additional keyword arguments passed to the underlying IK pose-check (e.g., tolerances).
        
        Returns
        -------
        joint_path : list or None
            A list of joint configurations corresponding to the planned Cartesian path,
            or None if planning failed or if any configuration is in self collision.
        """
        # Call the library's function with the proper inputs.
        joint_path = cartesian_motion_planning.plan_cartesian_motion(
            robot=self.robot.robotId,
            first_joint=self.robot.controllable_joint_idx[0],
            target_link=self.robot.end_effector_index,
            waypoint_poses=waypoint_poses,
            max_iterations=max_iterations,
            custom_limits=custom_limits,
            get_sub_conf=get_sub_conf,
            **kwargs
        )
        
        if joint_path is None:
            # print("No valid path found by plan_cartesian_motion.")
            return None

        # Verify that none of the configurations in the path are in self-collision.
        # Note: check_self_collision resets the robot's joints as part of the check.
        for config in joint_path:
            if len(self.robot.check_self_collision(config)) > 0:
                print("Self-collision detected in the planned path.")
                return None

        return joint_path