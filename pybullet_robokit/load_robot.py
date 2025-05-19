import numpy as np
from collections import defaultdict
import re
import time


class LoadRobot:
    def __init__(self, con, robot_urdf_path: str, start_pos, start_orientation, home_config, collision_objects=None, ee_link_name="tool0") -> None:
        """ Robot loader class

        Args:
            con (class): PyBullet client - an instance of the started env
            robot_urdf_path (str): filename/path to urdf file of robot
            start_pos (float list):  starting origin
            start_orientation (float list): starting orientation as a quaternion
        """
        assert isinstance(robot_urdf_path, str)

        self.con = con
        self.robot_urdf_path = robot_urdf_path
        self.start_pos = start_pos
        self.start_orientation = start_orientation
        self.home_config = home_config
        self.robotId = None
        self.home_ee_pos = None
        self.home_ee_ori = None
        self.collision_objects = collision_objects
        self.ee_link_name = ee_link_name

        # For tracking disabled collision pairs
        self._disabled_pairs = set()

        # will hold groups of revolute indices, *and* the following fixed joint idx
        self.spherical_groups = {}       # key = first revolute idx, value = fixed_joint_idx
        self.spherical_joint_idx = []    # list of [ix, iy, iz] triples
        
        self.setup_robot()

    def setup_robot(self):
        """ Initialize robot
        """
        assert self.robotId is None
        flags = self.con.URDF_USE_SELF_COLLISION | self.con.URDF_USE_INERTIA_FROM_FILE

        self.robotId = self.con.loadURDF(self.robot_urdf_path, self.start_pos, self.start_orientation, useFixedBase=True, flags=flags)
        self.num_joints = self.con.getNumJoints(self.robotId)

        self.end_effector_index = self.get_end_effector_index(self.ee_link_name)

        # All controllable revolute/prismatic joints
        self.controllable_joint_idx = []
        self.controllable_joint_types = []

        for joint in range(self.num_joints):
            joint_info = self.con.getJointInfo(self.robotId, joint)
            if joint_info[2] in {self.con.JOINT_REVOLUTE, self.con.JOINT_PRISMATIC}:
                self.controllable_joint_idx.append(joint_info[0])
                self.controllable_joint_types.append(joint_info[2])

        # Detect spherical groups *and* record the fixed joint after them
        self._disable_spherical_joint_collisions()
        self.disable_sequential_link_collisions(prefix='link')
        
        # Extract joint limits from urdf
        self.joint_limits = [self.con.getJointInfo(self.robotId, i)[8:10] for i in self.controllable_joint_idx]
        self.lower_limits = [t[0] for t in self.joint_limits]
        self.upper_limits = [t[1] for t in self.joint_limits]
        self.joint_ranges = [upper - lower for lower, upper in zip(self.lower_limits, self.upper_limits)]

        # Set the home position
        if self.home_config is None:
            self.home_config = [0.0] * len(self.controllable_joint_idx)

        self.zero_vec = [0.0] * len(self.controllable_joint_idx)

        self.reset_joint_positions(self.home_config)

        # Get the starting end-effector pos
        self.home_ee_pos, self.home_ee_ori = self.get_link_state(self.end_effector_index)

    def disable_collision_pair(self, link_a, link_b):
        """
        Disable collision between two links by their indices and record the pair.
        """
        pair = tuple(sorted((link_a, link_b)))
        self.con.setCollisionFilterPair(
            self.robotId, self.robotId,
            linkIndexA=pair[0],
            linkIndexB=pair[1],
            enableCollision=0
        )
        self._disabled_pairs.add(pair)

    def get_link_name_map(self, body_id):
        """Builds a map from linkIndex → linkName (and index −1 → base link)."""
        name_map = {}
        base_name = self.con.getBodyInfo(body_id)[0].decode('utf-8')
        name_map[-1] = base_name
        for ji in range(self.con.getNumJoints(body_id)):
            info = self.con.getJointInfo(body_id, ji)
            link_name = info[12].decode('utf-8')
            name_map[ji] = link_name
        return name_map

    def detect_all_self_collisions(self, body_id=None):
        """
        Loops through every (linkA, linkB) pair in the robot and reports overlaps,
        skipping any pairs that have been disabled.
        """
        if body_id is None:
            body_id = self.robotId

        name_map = self.get_link_name_map(body_id)
        num_j = self.con.getNumJoints(body_id)
        found = False

        # Step the sim to update collisions
        self.con.stepSimulation()

        for ia in range(-1, num_j):
            for ib in range(ia + 1, num_j):
                pair = (ia, ib)
                if pair in self._disabled_pairs:
                    continue

                pts = self.con.getClosestPoints(
                    bodyA=body_id,
                    bodyB=body_id,
                    distance=0.0,
                    linkIndexA=ia,
                    linkIndexB=ib
                )
                if pts:
                    # print(f"❌ Penetration between '{name_map[ia]}' and '{name_map[ib]}'")
                    found = True

        if not found:
            # print("✅ No self-collisions detected among any enabled link pairs.")
            pass

    def print_robot_environment_contacts(self, obstacle_ids):
        """
        For each obstacle in obstacle_ids, print any contact
        points between the robot (any link) and that obstacle.
        """
        name_map = self.get_link_name_map(self.robotId)
        for obs_id in obstacle_ids:
            contacts = self.con.getContactPoints(bodyA=self.robotId, bodyB=obs_id)
            if not contacts:
                continue
            # print(f"\n🤝 Contacts with obstacle body {obs_id}:")
            seen = set()
            for c in contacts:
                link_idx = c[3]  # linkIndexA
                if link_idx in seen:
                    continue
                seen.add(link_idx)
                link_name = name_map.get(link_idx, f"link{link_idx}")
                # print(f"  • '{link_name}' (link {link_idx}) touches obstacle {obs_id}")

    def _disable_spherical_joint_collisions(self):
        """
        Detect revolute triplets named _x_N, _y_N, _z_N, find the fixed joint immediately after them,
        and disable collisions between the parent of _x_N and that fixed link.
        """
        p = self.con
        base_name = p.getBodyInfo(self.robotId)[0].decode()

        # 1) Gather joint & link info
        joint_info = {}
        link_name_to_idx = {base_name: -1}

        for j in range(self.num_joints):
            info = p.getJointInfo(self.robotId, j)
            name = info[1].decode()
            jtype = info[2]
            parent_idx = info[16]
            child_name = info[12].decode()
            link_name_to_idx[child_name] = j
            parent_name = next(
                (n for n, idx in link_name_to_idx.items() if idx == parent_idx),
                base_name
            )
            joint_info[name] = {
                'idx': j,
                'type': jtype,
                'parent_link': parent_name,
                'child_link': child_name,
            }

        # 2) Collect revolute triplets
        suffix_map = defaultdict(lambda: {'x':None, 'y':None, 'z':None})
        pat = re.compile(r'^(.*)_(x|y|z)_(\d+)$')

        for name, data in joint_info.items():
            if data['type'] == p.JOINT_REVOLUTE:
                m = pat.match(name)
                if m:
                    axis, num = m.group(2), int(m.group(3))
                    suffix_map[num][axis] = name

        # 3) Disable collisions for each triplet
        self.spherical_joint_idx = []
        self.spherical_groups = {}

        for N, axes in suffix_map.items():
            if not all(axes.values()):
                continue

            ix = joint_info[axes['x']]['idx']
            iy = joint_info[axes['y']]['idx']
            iz = joint_info[axes['z']]['idx']
            self.spherical_joint_idx.append([ix, iy, iz])

            after_z = joint_info[axes['z']]['child_link']
            fixed_idx = next(
                (dat['idx'] for nm, dat in joint_info.items()
                 if dat['type'] == p.JOINT_FIXED and dat['parent_link'] == after_z),
                None
            )
            if fixed_idx is None:
                raise RuntimeError(f"Could not find fixed joint after '{after_z}' (sph #{N})")

            self.spherical_groups[ix] = fixed_idx
            
            before_x = joint_info[axes['x']]['parent_link']
            i0 = link_name_to_idx.get(before_x, -1)
            if i0 >= 0:
                self.disable_collision_pair(i0, fixed_idx)
            else:
                print(f"[warn] couldn't disable collision {before_x}↔{fixed_idx}")

    def disable_sequential_link_collisions(self, prefix='link'):
        """
        Disable collisions between link0↔link1, link1↔link2, etc.,
        based on the numeric suffix of your link names.
        """
        p = self.con

        # Build name→idx map
        base_name = p.getBodyInfo(self.robotId)[0].decode()
        link_name_to_idx = {base_name: -1}
        for j in range(self.num_joints):
            child_name = p.getJointInfo(self.robotId, j)[12].decode()
            link_name_to_idx[child_name] = j

        # Filter names with prefix+number
        pat = re.compile(rf'^{re.escape(prefix)}(\d+)$')
        numbered = []
        for name, idx in link_name_to_idx.items():
            m = pat.match(name)
            if m:
                numbered.append((int(m.group(1)), name))

        numbered.sort(key=lambda x: x[0])

        for (_, nameA), (_, nameB) in zip(numbered, numbered[1:]):
            idxA = link_name_to_idx[nameA]
            idxB = link_name_to_idx[nameB]
            self.disable_collision_pair(idxA, idxB)
            # print(f"Disabled collision: {nameA} (idx {idxA}) ↔ {nameB} (idx {idxB})")

    def get_end_effector_index(self, target_names=None):
        """
        Returns the first joint index whose link name matches one of the target_names.
        """
        if target_names is None:
            target_names = ['end_effector', 'ee_link', 'gripper_link', 'tool0']
        for i in range(self.num_joints):
            link_name = self.con.getJointInfo(self.robotId, i)[12].decode('utf-8')
            if link_name in target_names:
                return i
        return None

    def set_joint_positions(self, joint_positions):
        joint_positions = list(joint_positions)
        for i, joint_idx in enumerate(self.controllable_joint_idx):
            self.con.setJointMotorControl2(self.robotId, joint_idx, self.con.POSITION_CONTROL, joint_positions[i])
        self.con.stepSimulation()
    
    def set_joint_configuration(self, joint_positions):
        joint_positions = list(joint_positions)
        self.con.setJointMotorControlArray(self.robotId, self.controllable_joint_idx, self.con.POSITION_CONTROL, targetPositions=joint_positions, positionGains=[0.01]*len(joint_positions), velocityGains=[0.5]*len(joint_positions))
        self.con.stepSimulation()

    def reset_joint_positions(self, joint_positions=None):
        if joint_positions is None:
            joint_positions = self.home_config
        joint_positions = list(joint_positions)
        for i, joint_idx in enumerate(self.controllable_joint_idx):
            self.con.resetJointState(self.robotId, joint_idx, joint_positions[i])
        self.con.performCollisionDetection()  # Ensures up-to-date contact info
        self.con.stepSimulation()

    def set_joint_path(self, joint_path, delay=0.01):
        # Vizualize the interpolated positions
        for config in joint_path:
            self.set_joint_configuration(config)
            time.sleep(delay)  # Add a small delay for visualization purposes

    def get_joint_positions(self):
        return [self.con.getJointState(self.robotId, i)[0] for i in self.controllable_joint_idx]
    
    def get_link_state(self, link_idx):
        link_state = self.con.getLinkState(self.robotId, link_idx)
        link_position = np.array(link_state[0])
        link_orientation = np.array(link_state[1])
        return link_position, link_orientation
    
    def check_self_collision(self, joint_config):
        # Set the joint state and step the simulation
        self.reset_joint_positions(joint_config)

        # Return collision bool
        return len(self.con.getContactPoints(bodyA=self.robotId, bodyB=self.robotId)) > 0
     
    def check_collision_aabb(self, robot_id, plane_id):
        # Get AABB for the plane (ground)
        plane_aabb = self.con.getAABB(plane_id)

        # Iterate over each link of the robot
        for i in self.controllable_joint_idx:
            link_aabb = self.con.getAABB(robot_id, i)
            
            # Check for overlap between AABBs
            if (link_aabb[1][0] >= plane_aabb[0][0] and link_aabb[0][0] <= plane_aabb[1][0] and
                link_aabb[1][1] >= plane_aabb[0][1] and link_aabb[0][1] <= plane_aabb[1][1] and
                link_aabb[1][2] >= plane_aabb[0][2] and link_aabb[0][2] <= plane_aabb[1][2]):
                return True

        return False
    
    def collision_check(self, id_a, collision_objects=[]):
        if not collision_objects:
            collision_objects = self.collision_objects

        for obj in collision_objects:
            # Check for collision between the robot and the object
            collision = self.con.getContactPoints(bodyA=id_a, bodyB=obj)
            if len(collision) > 0:
                return True
        return False
    
    def inverse_kinematics(self, pose, pos_tol=1e-4, rest_config=None, max_iter=100, resample=1, num_resample=5):
        # Set the rest configuration to home if not provided
        rest_config = rest_config or self.home_config

        # Check if the pose is a list of length 2 or 3 (position or position + orientation)
        position, orientation = pose if len(pose) == 2 else (pose, None)
        
        # Stage IK arguments
        kwargs = {
            "lowerLimits": self.lower_limits,
            "upperLimits": self.upper_limits,
            "jointRanges": self.joint_ranges,
            "restPoses": rest_config,
            "residualThreshold": pos_tol,
            "maxNumIterations": max_iter
        }
    
        for _ in range(num_resample):
            if orientation is not None:
                joint_positions = self.con.calculateInverseKinematics(self.robotId, self.end_effector_index, position, orientation, **kwargs)
            else:
                joint_positions = self.con.calculateInverseKinematics(self.robotId, self.end_effector_index, position, **kwargs)

            if joint_positions is None:
                continue
            
            # Check for self-collision
            self.reset_joint_positions(joint_positions)
            if not self.check_self_collision(joint_positions):
                return joint_positions

            # Adjust rest configuration slightly to explore new solutions
            # Modify the rest configuration slightly instead of random sampling
            rest_config = np.clip(np.array(rest_config) + np.random.uniform(-0.05, 0.05, len(rest_config)),
                                self.lower_limits, self.upper_limits).tolist()
        return joint_positions
    
    def inverse_dynamics(self, joint_positions, joint_velocities=None, joint_accelerations=None):
        joint_velocities = joint_velocities or [0.0] * len(joint_positions)
        joint_accelerations = joint_accelerations or [0.0] * len(joint_positions)
        return self.con.calculateInverseDynamics(self.robotId, list(joint_positions), list(joint_velocities), list(joint_accelerations))

    def quaternion_angle_difference(self, q1, q2):
        # Compute the quaternion representing the relative rotation
        q1_conjugate = q1 * np.array([1, -1, -1, -1])  # Conjugate of q1
        q_relative = self.con.multiplyTransforms([0, 0, 0], q1_conjugate, [0, 0, 0], q2)[1]
        # The angle of rotation (in radians) is given by the arccos of the w component of the relative quaternion
        angle = 2 * np.arccos(np.clip(q_relative[0], -1.0, 1.0))
        return angle
    
    def check_pose_within_tolerance(self, current_pos, current_ori, target_pos, target_ori, tol=0.01):
        pos_err_axis = np.array(target_pos) - np.array(current_pos)
        pos_err_norm = np.linalg.norm(pos_err_axis)

        if target_ori is None:
            return pos_err_norm < tol, pos_err_axis, pos_err_norm, None, None
        
        ori_err = self.con.getDifferenceQuaternion(current_ori, target_ori)
        ori_err_axis, ori_err_angle = self.con.getAxisAngleFromQuaternion(ori_err)
        return pos_err_norm < tol and abs(ori_err_angle) < tol, pos_err_axis, pos_err_norm, ori_err_axis, ori_err_angle
    
    def is_pose_reachable(self, target_pose, tol=0.01, max_iter=200):
        # Check if the pose is a list of length 2 or 3 (position or position + orientation)
        target_pos, target_ori = target_pose if len(target_pose) == 2 else (target_pose, None)

        # Try solving IK
        joint_config = self.inverse_kinematics(target_pose, pos_tol=tol, max_iter=max_iter)
        if joint_config is None:
            return False, self.home_config
        
        # Check if resultant config puts end-effector within tolerance
        self.reset_joint_positions(joint_config)
        current_pos, current_ori = self.get_link_state(self.end_effector_index)

        pose_ok, *_ = self.check_pose_within_tolerance(current_pos, current_ori, target_pos, target_ori, tol)
        return pose_ok, joint_config
    
    def get_jacobian(self, joint_positions, local_pos=[0, 0, 0]):
        """
        Computes the full Jacobian matrix for the given joint positions and a specified local position.

            joint_positions (list or array-like): The list of joint angles/positions for the robot's joints.
            local_pos (list, optional): A 3D position vector [x, y, z] in the local frame of the end-effector. 
                        Defaults to [0, 0, 0].

            np.ndarray: A 6xN Jacobian matrix, where N is the number of joints. The matrix consists of:
                - The translational Jacobian (3xN) stacked on top of
                - The rotational Jacobian (3xN).
                If the Jacobian cannot be computed, a zero matrix of shape (6, N) is returned.
        """
        joint_positions = list(joint_positions)

        jac_t, jac_r = self.con.calculateJacobian(
            self.robotId, self.end_effector_index, local_pos, joint_positions, self.zero_vec, self.zero_vec
        )

        # Ensure correct stacking of the Jacobian components
        if jac_t is None or jac_r is None or len(jac_t) == 0 or len(jac_r) == 0:
            return np.zeros((6, len(joint_positions)))  # Return zero matrix if Jacobian is invalid

        jacobian = np.vstack((jac_t, jac_r))

        return jacobian
    
    def jacobian_viz(self, jacobian, end_effector_pos):
        # Visualization of the Jacobian columns
        num_columns = jacobian.shape[1]
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]  # Different colors for each column
        for i in range(num_columns):
            vector = jacobian[:, i]
            start_point = end_effector_pos
            end_point = start_point + 0.3 * vector[:3]  # Scale the vector for better visualization
            self.con.addUserDebugLine(start_point, end_point, colors[i % len(colors)], 2)

    def calculate_manipulability(self, joint_positions):
        jacobian = self.get_jacobian(joint_positions)
        return np.sqrt(np.linalg.det(jacobian @ jacobian.T))
    
    def safe_manipulability(self, joint_positions):
        J = self.get_jacobian(joint_positions)
        JJ_T = J @ J.T
        det = np.linalg.det(JJ_T)
        return np.sqrt(np.clip(det, 0.0, None))