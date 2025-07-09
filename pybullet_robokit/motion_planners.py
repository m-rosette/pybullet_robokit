import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
import pybullet_planning as pp
from pybullet_planning import (rrt_connect, get_distance_fn, get_sample_fn, get_extend_fn, get_collision_fn)
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
        # Ensure inputs are numpy arrays for vectorized operations
        start_configuration = np.array(start_configuration)
        end_configuration = np.array(end_configuration)

        # Normalize both configurations to [-pi, pi]
        start_configuration = (start_configuration + np.pi) % (2 * np.pi) - np.pi
        end_configuration = (end_configuration + np.pi) % (2 * np.pi) - np.pi

        # Calculate the angular difference between start and end configurations
        angular_difference = end_configuration - start_configuration

        # Wrap the angular difference to be within [-pi, pi]
        adjusted_difference = (angular_difference + np.pi) % (2 * np.pi) - np.pi

        # Check both paths: (end - start) and (end - start - 2*pi)
        adjusted_end_minus_2pi = end_configuration - 2 * np.pi
        adjusted_end_plus_2pi = end_configuration + 2 * np.pi

        # Compute all possible angular differences
        diff_original = adjusted_difference
        diff_minus_2pi = (adjusted_end_minus_2pi - start_configuration + np.pi) % (2 * np.pi) - np.pi
        diff_plus_2pi = (adjusted_end_plus_2pi - start_configuration + np.pi) % (2 * np.pi) - np.pi

        # Choose the smallest angular difference for each joint
        shortest_difference = np.where(np.abs(diff_minus_2pi) < np.abs(diff_original), diff_minus_2pi, diff_original)
        shortest_difference = np.where(np.abs(diff_plus_2pi) < np.abs(shortest_difference), diff_plus_2pi, shortest_difference)

        # Adjust the end configuration based on the shortest angular difference
        adjusted_end_configuration = start_configuration + shortest_difference

        return adjusted_end_configuration

    def interpolate_joint_trajectory(self, start_config, end_config, num_steps):
        """
        Interpolates a joint joint trajectory from start_config to end_config

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

        # Loop over each joint to interpolate using minimal angular changes
        for j in range(len(start_config)):
            # Interpolate linearly between the start and adjusted end angles
            for i in range(num_steps):
                interpolated_value = np.linspace(start_config[j], end_config[j], num_steps)[i]
                
                # Ensure the joint value stays within [-2pi, 2pi] after interpolation
                interpolated_value = np.clip(interpolated_value, self.robot.lower_limits[j], self.robot.upper_limits[j])
                
                # Store the interpolated value in the array
                interpolated_configs[i, j] = interpolated_value

        # Check for collisions in the interpolated path
        collision_in_path = any(self.robot.check_self_collision(config) for config in interpolated_configs)

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
            self.robot.reset_joint_positions(q_new.tolist())

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

    def peck_traj_gen(self, start_config, start_pose, end_config, end_pose, num_steps):
        """ Interpolates a joint trajectory between a start and end joint configuration, but also has the end effectory pass through a point that 
        is between the start and end effector position and orientation, while keeping the depth (y) at the same position as the start position.

        Args:
            start_config (np.array): starting joint coinfiguration
            start_pose (np.array): starting end effector pose [x, y, z, rz, ry, rz, w]
            end_config (np.array): ending joint coinfiguration
            end_pose (np.array): ending end effector pose [x, y, z, rz, ry, rz, w]
            num_steps (int): number of joint configurations in trajectory

        Returns:
            traj (np.array): joint trajectory
            collision (bool): describes any collisions in the trajectory
        """
        start_position = start_pose[:3]
        start_orientation = start_pose[3:]

        end_position = end_pose[:3]
        end_orientation = end_pose[3:]

        mid_position = np.copy(end_position)
        mid_position[1] = start_position[1]

        rotations = R.from_quat([start_orientation, end_orientation])

        # Define key times (e.g., t=0 for start, t=1 for end)
        times = np.array([0, 1])

        # Create SLERP object with two rotations
        slerp = Slerp(times, rotations)

        # Interpolate at t = 0.5 (midpoint)
        mid_rotation = slerp(0.5)

        # Get the quaternion for the mid rotation
        mid_orientation = mid_rotation.as_quat()

        test_rest_ee_pos = (start_position + mid_position) / 2 
        test_rest_config = self.robot.inverse_kinematics(test_rest_ee_pos, mid_orientation, rest_config=list(start_config))
        test_rest_config[4:] = self.shortest_angular_distance(start_config[4:], end_config[4:])

        mid_config = self.robot.inverse_kinematics(mid_position, mid_orientation, rest_config=list(test_rest_config))

        first_half_traj, path_collision1 = self.interpolate_joint_trajectory(start_config, mid_config, int(num_steps/2))
        second_half_traj, path_collision2 = self.interpolate_joint_trajectory(first_half_traj[-1], end_config, int(num_steps/2))

        traj = np.vstack((first_half_traj, second_half_traj))

        if path_collision1 or path_collision2:
            collision = True
        else:
            collision = False

        return traj, collision
    
    def peck_traj_gen2(self, start_config, start_pose, end_config, end_pose, num_steps):
        """ Interpolates a joint trajectory between a start and end joint configuration, but also has the end effectory pass through multiple points 
        that are between the start and end effector position and orientation, while keeping the depth (y) at the same position as the start position.

        Args:
            start_config (np.array): starting joint coinfiguration
            start_pose (np.array): starting end effector pose [x, y, z, rz, ry, rz, w]
            end_config (np.array): ending joint coinfiguration
            end_pose (np.array): ending end effector pose [x, y, z, rz, ry, rz, w]
            num_steps (int): number of joint configurations in trajectory

        Returns:
            traj (np.array): joint trajectory
            collision (bool): describes any collisions in the trajectory
        """
        start_position = start_pose[:3]
        start_orientation = start_pose[3:]

        end_position = end_pose[:3]
        end_orientation = end_pose[3:]

        mid_position = np.copy(end_position)
        mid_position[1] = start_position[1]

        num_midpoints = 3
        mid_positions = np.linspace(start_position, mid_position, num_midpoints + 2)
        # mid_positions = np.linspace(start_position, end_position, num_midpoints + 2)
        mid_positions[:, 1] = start_position[1]

        rotations = R.from_quat([start_orientation, end_orientation])

        # Define key times (e.g., t=0 for start, t=1 for end)
        times = np.array([0, 1])

        # Create SLERP object with two rotations
        slerp = Slerp(times, rotations)

        # Interpolate at t = 0.5 (midpoint)
        mid_rotation0 = slerp(0.25)
        mid_rotation1 = slerp(0.5)
        mid_rotation2 = slerp(0.75)

        # Get the quaternion for the mid rotation
        mid_orientation0 = mid_rotation0.as_quat()
        mid_orientation1 = mid_rotation1.as_quat()
        mid_orientation2 = mid_rotation2.as_quat()

        mid_config0 = self.robot.inverse_kinematics(mid_positions[1], mid_orientation0, rest_config=list(start_config))
        first_traj, path_collision0 = self.interpolate_joint_trajectory(start_config, mid_config0, int(num_steps/6))

        mid_config1 = self.robot.inverse_kinematics(mid_positions[2], mid_orientation1, rest_config=list(first_traj[-1]))
        second_traj, path_collision1 = self.interpolate_joint_trajectory(first_traj[-1], mid_config1, int(num_steps/6))

        mid_config2 = self.robot.inverse_kinematics(mid_positions[3], mid_orientation2, rest_config=list(second_traj[-1]))
        third_traj, path_collision2 = self.interpolate_joint_trajectory(second_traj[-1], mid_config2, int(num_steps/6))

        fourth_traj, path_collision3 = self.interpolate_joint_trajectory(third_traj[-1], end_config, int(num_steps/2))

        traj = np.vstack((first_traj, second_traj, third_traj, fourth_traj))

        if path_collision0 or path_collision1 or path_collision2 or path_collision3:
            collision = True
        else:
            collision = False

        return traj, collision
    
    def task_space_path_interp(self, start_config, start_pose, end_config, end_pose, num_steps):
        """ Interpolates a joint trajectory between a start and end joint configuration, where all intermediate configurations
        are generated in task space while maintaining a desired start and end joint configuration.

        Args:
            start_config (np.array): starting joint coinfiguration
            start_pose (np.array): starting end effector pose [x, y, z, rz, ry, rz, w]
            end_config (np.array): ending joint coinfiguration
            end_pose (np.array): ending end effector pose [x, y, z, rz, ry, rz, w]
            num_steps (int): number of joint configurations in trajectory

        Returns:
            interpolated_configs (np.array): joint trajectory
            collision_in_path (bool): describes any collisions in the trajectory
        """
        start_position = start_pose[:3]
        start_orientation = start_pose[3:]

        end_position = end_pose[:3]
        end_orientation = end_pose[3:]

        rotations = R.from_quat([start_orientation, end_orientation])

        # Define key times (e.g., t=0 for start, t=1 for end)
        times = np.array([0, 1])

        # Create SLERP object with two rotations
        slerp = Slerp(times, rotations)

        # Split the number of steps for each phase
        num_steps_phase1 = num_steps // 2  # First half for x and z translation
        num_steps_phase2 = num_steps - num_steps_phase1  # Second half for y translation

        # Create an empty array for storing interpolated joint configurations
        interpolated_configs = np.zeros((num_steps, len(start_config)))
        interpolated_configs[0] = start_config

        # Phase 1: Interpolate x and z first, keep y constant
        for i in range(1, num_steps_phase1):
            # Interpolate x and z linearly between the start and end pose
            interp_x = np.linspace(start_position[0], end_position[0], num_steps_phase1)[i]
            interp_z = np.linspace(start_position[2], end_position[2], num_steps_phase1)[i]

            # Keep y constant as the start y value
            interp_y = start_position[1]

            intermediate_position = np.array([interp_x, interp_y, interp_z])

            # Interpolate orientation using SLERP
            t = i / num_steps_phase1
            intermediate_orientation = slerp(t).as_quat()  # Get interpolated orientation

            # Create intermediate pose (x, y, z) + maintain original orientation
            intermediate_pose = np.concatenate((intermediate_position, intermediate_orientation))

            # Compute joint configuration for this pose using inverse kinematics
            joint_config = self.robot.inverse_kinematics(intermediate_position, intermediate_orientation, rest_config=list(interpolated_configs[i-1]))
            joint_config = self.shortest_angular_distance(interpolated_configs[i-1], joint_config)

            interpolated_configs[i] = joint_config

        # Phase 2: Interpolate y while keeping x and z fixed at their final values
        for i in range(num_steps_phase2):
            # Interpolate y linearly between the intermediate y and end y
            interp_y = np.linspace(start_position[1], end_position[1], num_steps_phase2)[i]

            # Keep x and z fixed at their final values
            interp_x = end_position[0]
            interp_z = end_position[2]

            # Create final pose (x, y, z) + final orientation
            final_position = np.array([interp_x, interp_y, interp_z])
            
            # Interpolate orientation using SLERP
            t = i / num_steps_phase2
            intermediate_orientation = slerp(0.5 + 0.5 * t).as_quat()  # Interpolate for second phase

            intermediate_pose = np.concatenate((final_position, intermediate_orientation))

            # Compute joint configuration for this pose using inverse kinematics
            joint_config = self.robot.inverse_kinematics(final_position, intermediate_orientation, rest_config=list(interpolated_configs[i-1]))
            joint_config = self.shortest_angular_distance(interpolated_configs[i-1], joint_config)
            interpolated_configs[num_steps_phase1 + i] = joint_config
        
        # Force the last configuration to match end_config
        interpolated_configs[-1] = end_config
        interpolated_configs[-1] = self.shortest_angular_distance(interpolated_configs[-2], end_config)

        # Check for collisions in the interpolated path
        collision_in_path = any(self.robot.check_self_collision(config) for config in interpolated_configs)

        return interpolated_configs, collision_in_path

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

    def vector_field_sample_fn(self, goal_position, alpha=0.8):
        def sample():
            random_conf = np.random.uniform([limit[0] for limit in self.robot.joint_limits], 
                                            [limit[1] for limit in self.robot.joint_limits])
            self.robot.set_joint_positions(random_conf)
            end_effector_position, _ = self.robot.get_link_state(self.robot.end_effector_index)
            
            vector_to_goal = np.array(goal_position) - end_effector_position
            guided_position = end_effector_position + vector_to_goal
            # guided_conf = np.array(self.robot.robot.inverse_kinematics(guided_position, goal_orientation))
            guided_conf = np.array(self.robot.inverse_kinematics(guided_position))
            final_conf = (1 - alpha) * random_conf + alpha * guided_conf
            
            return final_conf
        return sample
    
    def make_strict_collision_fn(self, obstacles):
        def fn(q):
            # 1) move into q
            self.robot.set_joint_configuration(q)
            # 2) collision check
            #  a) self-collision?
            if self.robot.check_self_collision(q):
                self.robot.detect_all_self_collisions(self.robot.robotId)
                return True
            #  b) environment collision?
            if self.robot.collision_check(self.robot.robotId, obstacles):
                self.robot.detect_all_self_collisions(self.robot.robotId)
                return True
            return False
        return fn

    def rrt_path(self, start_joint_config, end_joint_config, collision_objects=None, steps=None, rrt_iter=500):
        extend_fn = get_extend_fn(self.robot.robotId, self.robot.controllable_joint_idx)
        # collision_fn = get_collision_fn(self.robot.robotId, self.robot.controllable_joint_idx, collision_objects)
        collision_fn = self.make_strict_collision_fn(collision_objects)
        distance_fn = get_distance_fn(self.robot.robotId, self.robot.controllable_joint_idx)
        sample_fn = get_sample_fn(self.robot.robotId, self.robot.controllable_joint_idx)
        # sample_fn = self.vector_field_sample_fn(target_pos)

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