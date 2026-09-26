import os
import sys
import glob
import queue
import select
import termios
import threading
import tty
import argparse
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from pybullet_robokit.pyb_utils import PybUtils
from pybullet_robokit.load_objects import LoadObjects
from pybullet_robokit.load_robot import LoadRobot
from pybullet_robokit.motion_planners import KinematicChainMotionPlanner


def get_urdf_path(user_input, default_dir, default_file):
    """
    Determines the correct URDF file path.
    
    Args:
        user_input (str): The user-provided URDF path or filename.
        default_dir (str): The default directory where URDF files are stored.
        default_file (str): The default URDF filename.
    
    Returns:
        str: The resolved URDF file path.
    """
    if os.path.isabs(user_input) and os.path.isfile(user_input):
        return user_input
    
    potential_path = os.path.join(default_dir, user_input)
    if os.path.isfile(potential_path):
        return potential_path
    
    return os.path.join(default_dir, default_file)


class ViewRobot:
    def __init__(self, robot_urdf_path: str, robot_home_pos, ik_tol=0.01, renders=True, ee_link_name='ee_link'):
        """
        Initialize the ViewRobot class.

        Args:
            robot_urdf_path (str): Path to the URDF file of the robot.
            robot_home_pos (list): Home position of the robot joints.
            ik_tol (float, optional): Tolerance for inverse kinematics. Defaults to 0.01.
            renders (bool, optional): Whether to visualize the robot in the PyBullet GUI. Defaults to True.
        """
        self.pyb = PybUtils(renders=renders)
        self.object_loader = LoadObjects(self.pyb.con)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_dir = os.path.join(script_dir, 'urdf', 'trees')
        flags = 0 #self.pyb.con.URDF_MERGE_FIXED_LINKS
        self.tree_id = self.object_loader.load_urdf(os.path.join(urdf_dir, "v_trellis_template.urdf"),
                                        start_pos=[0, 1, 0], 
                                        start_orientation=[0, 0, 0], 
                                        fix_base=True,
                                        flags=flags)
        self.object_loader.collision_objects.append(self.tree_id)

        self.robot = LoadRobot(self.pyb.con, 
                               robot_urdf_path, 
                               [0, 0, 0], 
                               self.pyb.con.getQuaternionFromEuler([0, 0, 0]), 
                               robot_home_pos, 
                               collision_objects=self.object_loader.collision_objects,
                               ee_link_name=ee_link_name)

        self.ik_tol = ik_tol

    def cartesian_path_test(self):
        # Euler angles to point the end-effector in the y-direction
        orientation = list(self.pyb.con.getQuaternionFromEuler([-np.pi/2, 0, 0]))

        # Define start and end positions
        start_pos = [0.5, 0.75, 0.5]
        end_pos = [-0.5, 0.75, 0.5]

        # Create start and end poses as (position, orientation)
        start_pose = (start_pos, orientation)
        end_pose = (end_pos, orientation)

        print("Start Pose:", start_pose)
        print("End Pose:", end_pose)

        # Interpolate a list of waypoint poses between start_pose and end_pose
        num_points = 10
        waypoints = []
        for t in np.linspace(0, 1, num_points):
            interp_pos = (np.array(start_pos) * (1 - t) + np.array(end_pos) * t).tolist()

            # Orientation remains constant
            waypoints.append((interp_pos, orientation))

        # Call the plan_cartesian_motion_path function with the list of waypoint poses
        joint_path = self.robot.plan_cartesian_motion_path(waypoints, max_iterations=10000)

        if joint_path is None:
            print("Cartesian path planning failed.")
            return

        # Execute the planned joint path
        self.robot.set_joint_path(joint_path)

    def test_resolved_rate_motion_control(self):
        # input("Press Enter to start the resolved rate motion control test...")
        target_pos = [0, 1.5, 1.5]
        motion_planner = KinematicChainMotionPlanner(self.robot)

        target_orientations = [
            np.array([180, 0, 90]), # top-down (-z)

            np.array([90, 0, 180]), # front-back (+y)

            np.array([0, 0, -90]), # bottom-up (+z)

            np.array([90, 0, -90]), # right-left (-x)

            np.array([90, 0, 180]), # front-back (+y)

            np.array([90, 0, 90]), # left-right (+x)
        ]

        for target_ori in target_orientations:
            target_ori = R.from_euler('xyz', target_ori, degrees=True).as_quat()
            
            q_final, manip_score, delta_joint_score, pose_error = motion_planner.resolved_rate_control(
                                                                        (target_pos, target_ori), 
                                                                        max_steps=400,
                                                                        plot_manipulability=False, 
                                                                        alpha=0.75,
                                                                        beta=0.75,
                                                                        damping_lambda=0.15, 
                                                                        manipulability_gain=0.1, 
                                                                        stall_vel_threshold=0.1, 
                                                                        stall_patience=10)
            print("\nManipulability score:", manip_score)
            print("Delta joint score:", delta_joint_score)
            print("Position error:", pose_error[0])
            print("Orientation error:", pose_error[1])
            print()

            # if q_final:
            #     self.robot.set_joint_configuration(q_final)
        
        print("Test complete. Press Ctrl+C to exit.")
        while True:
            self.pyb.con.stepSimulation()

    def rrt_path_test(self):
        # Start configuration is the robots home position
        start_config = self.robot.home_config

        # Get IK to the target position
        target_pos = [0.25, 0.5, 1.5]
        target_ori = np.array([90, 0, 180])
        target_ori = R.from_euler('xyz', target_ori, degrees=True).as_quat()
        target_pose = (target_pos, target_ori)
        target_config = self.robot.inverse_kinematics(target_pose, pos_tol=self.ik_tol, max_iter=1000, num_resample=10)

        # Initialize the motion planner
        motion_planner = KinematicChainMotionPlanner(self.robot)

        # Bias RRT-Connect away from large proximal-joint swings (higher weight on
        # shoulder/elbow, lower on wrist), then shortcut/smooth the raw path afterward
        # to remove RRT's characteristic detours and backtracking.
        num_joints = len(self.robot.controllable_joint_idx)
        joint_weights = np.linspace(num_joints, 1, num_joints)

        # Pass the start and target configurations to the RRT planner
        joint_path = motion_planner.rrt_path(start_config, target_config, rrt_iter=1000,
                                              collision_objects=self.object_loader.collision_objects,
                                              steps=500, joint_weights=joint_weights,
                                              smooth=True, smooth_iterations=150)

        # Check if the path is valid
        if joint_path is None:
            print("\nRRT path planning failed.\n")
            return
        else:
            print("\nRRT path planning succeeded.\n")

        path_cost = np.sum(np.linalg.norm(np.diff(np.array(joint_path), axis=0), axis=1))
        print(len(joint_path), "steps in the path")
        print(f"Total joint-space travel: {path_cost:.3f} rad\n")

        # Execute the planned joint path
        self.robot.set_joint_path(joint_path)

        print("Test complete. Press Ctrl+C to exit.")
        while True:
            self.pyb.con.stepSimulation()

    def test_collisions(self):
        # Define joint configuration for the robot
        # joint_config = [0, 0, 0, 0, 0, 0]
        joint_config = self.robot.home_config
        # joint_config = [0.0, 0.95]

        self.robot.set_joint_configuration(joint_config)
        self.robot.reset_joint_positions(joint_config)

        self.robot.detect_all_self_collisions(self.robot.robotId)
        self.robot.print_robot_environment_contacts(self.object_loader.collision_objects)

        # Check for collisions
        print('Self-collision check:')
        print('AABB: ', self.robot.check_collision_aabb(self.robot.robotId, self.robot.robotId))
        print('General: ', self.robot.collision_check(self.robot.robotId, collision_objects=self.object_loader.collision_objects))
        print('Self: ', self.robot.check_self_collision(self.robot.home_config))

        print("\nTest complete. Press Ctrl+C to exit.\n")
        while True:
            self.pyb.con.stepSimulation()

    def main(self):
        target_positions = np.random.uniform(low=[-2.0, -2.0, 0], high=[2.0, 2.0, 2.0], size=(20, 3)).tolist()
        target_point_id = None

        for i, position in enumerate(target_positions):
            input("Press Enter to continue...")

            if target_point_id is not None:
                # Remove the previous target point after reaching it
                self.pyb.con.removeBody(target_point_id) 
        
            target_point_id = self.object_loader.load_urdf("sphere2.urdf", 
                                        start_pos=position, 
                                        start_orientation=[0, 0, 0], 
                                        fix_base=True, 
                                        radius=0.05)
            
            # Set the robot to its home position
            self.robot.set_joint_configuration(self.robot.home_config)
        
            # Move arm to the target pose using inverse kinematics
            joint_config = self.robot.inverse_kinematics(position, pos_tol=self.ik_tol, max_iter=1000, num_resample=0)

            # if self.robot.check_collision_aabb(self.robot.robotId, self.robot.robotId):
            #     print("Collision detected!")
            # joint_path = self.robot.optimized_rrt_path(self.robot.home_config, joint_config)
            # if joint_path is None:
            #     print("RRT path planning failed. Defaulting to plain IK.")
            #     self.robot.set_joint_configuration(joint_config)

            # else:
            #     # Execute the planned joint path
            #     self.robot.set_joint_path(joint_path)
            self.robot.reset_joint_positions(joint_config)
            self.robot.set_joint_configuration(joint_config)

            # Step simulation and render
            for _ in range(240):  # Adjust number of simulation steps as needed
                self.pyb.con.stepSimulation()
                time.sleep(1./240.)  # Sleep to match real-time
            
        while True:
            self.pyb.con.stepSimulation()


def load_amiga_ur5e_env(renders=True, urdf_dir=None, robot_base_yaw=0.0, ee_link_name='tool0'):
    """ Loads the UR5e mounted on the Amiga, with the Amiga chassis and mounted hardware
    (slider, mast, GPS/Oak camera housing, Amiga Brain) as collision primitives.

    Args:
        renders (bool, optional): visualize the robot in the PyBullet GUI. Defaults to True.
        urdf_dir (str, optional): directory holding ur5e/ur5e.urdf and
            amiga/visual/frame_on_amiga_v2_simplified.stl. Defaults to this package's urdf/robots.
        robot_base_yaw (float, optional): rotation of the robot base about world z, in radians.
            Defaults to 0.
        ee_link_name (str, optional): end-effector link name. Defaults to 'tool0'.

    Returns:
        pyb (PybUtils): the PyBullet client wrapper
        object_loader (LoadObjects): holds the environment's collision_objects
        robot (LoadRobot): the loaded UR5e
    """
    if urdf_dir is None:
        urdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'urdf', 'robots')
    ur5e_urdf_path = os.path.join(urdf_dir, 'ur5e', 'ur5e.urdf')
    amiga_mesh_path = os.path.join(urdf_dir, 'amiga', 'visual', 'frame_on_amiga_v2_simplified.stl')

    pyb = PybUtils(renders=renders)
    object_loader = LoadObjects(pyb.con)

    # Amiga chassis + mounted hardware as simple collision primitives
    slider_id = pyb.con.createMultiBody(
        baseCollisionShapeIndex=pyb.con.createCollisionShape(pyb.con.GEOM_BOX, halfExtents=[0.6, 0.1, 0.075]),
        basePosition=[0.0, 0.29845, 0.96])
    mast_id = pyb.con.createMultiBody(
        baseCollisionShapeIndex=pyb.con.createCollisionShape(pyb.con.GEOM_BOX, halfExtents=[0.035, 0.035, 0.475]),
        basePosition=[0.57, -0.15, 1.03])
    gps_oak_id = pyb.con.createMultiBody(
        baseCollisionShapeIndex=pyb.con.createCollisionShape(pyb.con.GEOM_BOX, halfExtents=[0.075, 0.10, 0.10]),
        basePosition=[0.57, -0.15, 1.62])
    brain_id = pyb.con.createMultiBody(
        baseCollisionShapeIndex=pyb.con.createCollisionShape(pyb.con.GEOM_BOX, halfExtents=[0.145, 0.10, 0.0875]),
        basePosition=[0.57, -0.315, 1.25])
    amiga_frame_id = pyb.con.createMultiBody(
        baseCollisionShapeIndex=pyb.con.createCollisionShape(
            pyb.con.GEOM_MESH, fileName=amiga_mesh_path, flags=pyb.con.GEOM_FORCE_CONCAVE_TRIMESH),
        baseVisualShapeIndex=-1,
        basePosition=[0, 0, 0])
    object_loader.collision_objects.extend([amiga_frame_id, slider_id, mast_id, gps_oak_id, brain_id])

    robot_home_pos = [np.pi/4, -np.pi/2, 2*np.pi/3, 5*np.pi/6, -np.pi/2, 0]
    robot = LoadRobot(pyb.con,
                       ur5e_urdf_path,
                       [-0.092075, 0.29845, 1.04775],
                       pyb.con.getQuaternionFromEuler([0, 0, robot_base_yaw]),
                       robot_home_pos,
                       collision_objects=object_loader.collision_objects,
                       ee_link_name=ee_link_name)

    return pyb, object_loader, robot


def resolve_cached_paths_file(paths_file):
    """ Resolves a cached-trajectory .npy file, given either the file itself or a run directory
    (in which case the newest reachable_paths_merged_*.npy in it is used). """
    if os.path.isdir(paths_file):
        candidates = sorted(glob.glob(os.path.join(paths_file, 'reachable_paths_merged_*.npy')))
        if not candidates:
            raise FileNotFoundError(f"No reachable_paths_merged_*.npy found in {paths_file}")
        return candidates[-1]
    return paths_file


def find_matching_voxels_file(paths_file):
    """ Finds the reachable_voxels csv saved alongside a reachable_paths npy (same tag, e.g.
    '_merged' or '_w0003'). Its rows line up with the paths' last axis. Returns None if absent.

    The two files are saved separately, so their timestamps can differ by a second - match on
    the tag rather than the full name, and take the newest match.
    """
    stem = os.path.splitext(os.path.basename(paths_file))[0]
    tag = stem[len('reachable_paths'):-len('_YYYYmmdd_HHMMSS')]
    candidates = sorted(glob.glob(os.path.join(os.path.dirname(paths_file), f'reachable_voxels{tag}_*.csv')))
    return candidates[-1] if candidates else None


def view_cached_trajectories(paths_file, urdf_dir=None, robot_base_yaw=np.pi/4, ee_link_name='gripper_link',
                             index=None, delay=0.01):
    """ Loads trajectories cached by trajectory_cache (an npy of shape (num_configs, num_joints,
    num_paths)) and plays back the chosen one on the UR5e-on-Amiga in the PyBullet GUI.

    Indices are entered at a terminal prompt (read in the background, so the GUI stays live and a
    new index interrupts the current playback), or stepped with the Left/Right arrow keys in the GUI
    window or the terminal (NaN paths are skipped).
    The target voxel for the path (if its reachable_voxels csv is found) is drawn as a green sphere.

    Args:
        paths_file (str): reachable_paths*.npy file, or a run directory containing a merged one.
        urdf_dir (str, optional): see load_amiga_ur5e_env.
        robot_base_yaw (float, optional): must match the base yaw the paths were generated with.
            Defaults to pi/4 (what trajectory_cache's path_cache/parallel_cache use).
        ee_link_name (str, optional): end-effector link name. Defaults to 'gripper_link'.
        index (int, optional): path to play first. Defaults to None (wait for input).
        delay (float, optional): seconds between configs during playback. Defaults to 0.01.
    """
    paths_file = resolve_cached_paths_file(paths_file)
    paths = np.load(paths_file)
    num_paths = paths.shape[2]
    print(f"\nLoaded {num_paths} paths of {paths.shape[0]} configs x {paths.shape[1]} joints from {paths_file}")

    voxels_file = find_matching_voxels_file(paths_file)
    voxels = np.loadtxt(voxels_file).reshape(-1, 3) if voxels_file else None
    if voxels is not None and len(voxels) != num_paths:
        print(f"Warning: {voxels_file} has {len(voxels)} rows but there are {num_paths} paths; "
              f"not showing target voxels.")
        voxels = None

    valid = ~np.isnan(paths).any(axis=(0, 1))
    if not valid.any():
        print("No valid paths in this file.")
        return

    pyb, object_loader, robot = load_amiga_ur5e_env(renders=True, urdf_dir=urdf_dir,
                                                    robot_base_yaw=robot_base_yaw, ee_link_name=ee_link_name)
    target_visual = pyb.con.createVisualShape(pyb.con.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 0.8])
    target_id = None
    current = None
    joint_path = None
    frame = 0

    # Read terminal input on a background thread so the GUI stays live while waiting for an
    # index. Commands (typed lines, or 'next'/'prev' for arrow keys) go to the main loop via a queue.
    commands = queue.Queue()
    arrow_commands = {b'[C': 'next', b'[D': 'prev'}

    def read_keys():
        """ Terminal in cbreak mode: arrow keys act immediately, other keys build up a line
        (echoed by hand, since cbreak turns echo off) that is submitted on Enter. """
        fd = sys.stdin.fileno()
        buf = ''
        while True:
            ch = os.read(fd, 1)
            if ch in (b'', b'\x04'):  # stdin closed / Ctrl+D
                commands.put('q')
                return
            if ch == b'\x1b':
                # Arrow keys arrive as ESC [ C/D; a lone Esc has nothing following it
                seq = os.read(fd, 2) if select.select([fd], [], [], 0.05)[0] else b''
                if seq in arrow_commands:
                    buf = ''
                    commands.put(arrow_commands[seq])
            elif ch in (b'\n', b'\r'):
                sys.stdout.write('\n')
                sys.stdout.flush()
                commands.put(buf.strip().lower())
                buf = ''
            elif ch in (b'\x7f', b'\x08'):
                if buf:
                    buf = buf[:-1]
                    sys.stdout.write('\b \b')
                    sys.stdout.flush()
            else:
                c = ch.decode(errors='ignore')
                if c.isprintable():
                    buf += c
                    sys.stdout.write(c)
                    sys.stdout.flush()

    def read_lines():
        """ Fallback when stdin isn't a terminal: whole lines only. """
        for line in sys.stdin:
            line = line.strip().lower()
            commands.put(arrow_commands.get(line[1:].upper().encode(), line) if line.startswith('\x1b') else line)
        commands.put('q')

    old_term = None
    if sys.stdin.isatty():
        old_term = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        threading.Thread(target=read_keys, daemon=True).start()
    else:
        threading.Thread(target=read_lines, daemon=True).start()
    prompt = (f"Path index [0-{num_paths - 1}], Enter/'r' to replay, 'q' to quit, "
              f"Left/Right arrows to step: ")

    def step_valid(start, direction):
        """ Next index from `start` in `direction` (+1/-1) that has a valid path, wrapping around. """
        i = start
        for _ in range(num_paths):
            i = (i + direction) % num_paths
            if valid[i]:
                return i
        return start

    def play(i):
        nonlocal current, joint_path, frame, target_id
        if not valid[i]:
            print(f"Path {i} contains NaNs (no valid path was found for it); skipping.")
            return
        current = i
        joint_path = paths[:, :, current]
        frame = 0
        if target_id is not None:
            pyb.con.removeBody(target_id)
            target_id = None
        msg = f"Playing path {current}"
        if voxels is not None:
            target_id = pyb.con.createMultiBody(baseVisualShapeIndex=target_visual, basePosition=voxels[current])
            msg += f" to voxel {np.round(voxels[current], 3).tolist()}"
        print(msg)

    def step(direction):
        """ Plays the previous (-1) / next (+1) valid path. """
        start = current if current is not None else (-1 if direction > 0 else 0)
        play(step_valid(start, direction))

    if index is not None:
        if -num_paths <= index < num_paths:
            play(index % num_paths)
        else:
            print(f"Index {index} out of range [0, {num_paths - 1}]")
    print(prompt, end='', flush=True)

    try:
        while pyb.con.isConnected():
            # Terminal commands
            quit_requested = False
            while not commands.empty():
                reply = commands.get()
                if reply in ('q', 'quit', 'exit'):
                    quit_requested = True
                    break
                if reply in ('next', 'prev'):
                    print()
                    step(1 if reply == 'next' else -1)
                elif reply in ('r', ''):
                    if current is not None:
                        play(current)
                else:
                    try:
                        i = int(reply)
                    except ValueError:
                        print(f"Not an index: {reply!r}")
                    else:
                        if -num_paths <= i < num_paths:
                            play(i % num_paths)
                        else:
                            print(f"Index {i} out of range [0, {num_paths - 1}]")
                print(prompt, end='', flush=True)
            if quit_requested:
                break

            # GUI arrow keys step to the previous/next valid path
            keys = pyb.con.getKeyboardEvents()
            for key, direction in ((pyb.con.B3G_RIGHT_ARROW, 1), (pyb.con.B3G_LEFT_ARROW, -1)):
                if keys.get(key, 0) & pyb.con.KEY_WAS_TRIGGERED:
                    print()
                    step(direction)
                    print(prompt, end='', flush=True)

            # Advance playback one config per tick so new commands can interrupt it
            if joint_path is not None and frame < len(joint_path):
                robot.reset_joint_positions(joint_path[frame], step_sim=True)
                frame += 1
                time.sleep(delay)
            else:
                time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        if old_term is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_term)

    print()  # PybUtils disconnects on exit


def two_stage_cartesian_demo(renders=True, num_steps=150):
    """ Plans and executes a two-stage Cartesian path on the UR5e-on-Amiga: world X/Z motion
    (reach depth and height) followed by world Y motion (reach out to the side).
    """
    pyb, object_loader, robot = load_amiga_ur5e_env(renders=renders)
    motion_planner = KinematicChainMotionPlanner(robot)

    # Example target: reach forward/up in X/Z, then out to the side in Y
    start_config = robot.home_config
    start_position, start_orientation = robot.get_link_state(robot.end_effector_index)
    target_position = start_position + np.array([0.15, 0.2, 0.05])
    end_config = robot.inverse_kinematics((target_position, start_orientation), pos_tol=0.01,
                                           rest_config=start_config, max_iter=1000)

    joint_path, collision_in_path = motion_planner.two_stage_cartesian_path(
        start_config, end_config, num_steps=num_steps)

    print(f"\nTwo-stage Cartesian path: {len(joint_path)} steps, collision_in_path={collision_in_path}\n")

    robot.reset_joint_positions(start_config)
    robot.set_joint_path(joint_path)

    print("Test complete. Holding final pose. Press Ctrl+C to exit.")
    while True:
        robot.reset_joint_positions(joint_path[-1])
        pyb.con.stepSimulation()


def rrt_amiga_demo(renders=True, rrt_iter=1000, steps=500):
    """ Plans and executes an RRT-Connect path on the UR5e-on-Amiga, using the joint-weighted
    distance metric and shortcutting/smoothing pass from `KinematicChainMotionPlanner.rrt_path`
    to avoid the large, "wacky" joint swings plain RRT-Connect tends to produce.
    """
    pyb, object_loader, robot = load_amiga_ur5e_env(renders=renders)
    motion_planner = KinematicChainMotionPlanner(robot)

    # Same example target as the two-stage Cartesian demo, for an apples-to-apples comparison
    start_config = robot.home_config
    start_position, start_orientation = robot.get_link_state(robot.end_effector_index)
    target_position = start_position + np.array([0.15, 0.2, 0.05])
    end_config = robot.inverse_kinematics((target_position, start_orientation), pos_tol=0.01,
                                           rest_config=start_config, max_iter=1000)

    # Bias RRT-Connect away from large proximal-joint swings, then shortcut/smooth the raw path
    num_joints = len(robot.controllable_joint_idx)
    joint_weights = np.linspace(num_joints, 1, num_joints)

    joint_path = motion_planner.rrt_path(start_config, end_config, rrt_iter=rrt_iter,
                                          collision_objects=object_loader.collision_objects,
                                          steps=steps, joint_weights=joint_weights,
                                          smooth=True, smooth_iterations=150)

    if joint_path is None:
        print("\nRRT path planning failed.\n")
        return

    path_cost = np.sum(np.linalg.norm(np.diff(np.array(joint_path), axis=0), axis=1))
    print(f"\nRRT path planning succeeded: {len(joint_path)} steps, "
          f"total joint-space travel: {path_cost:.3f} rad\n")

    robot.reset_joint_positions(start_config)
    robot.set_joint_path(joint_path)

    print("Test complete. Holding final pose. Press Ctrl+C to exit.")
    while True:
        robot.reset_joint_positions(joint_path[-1])
        pyb.con.stepSimulation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View a robot in PyBullet simulation.")
    parser.add_argument("-u", "--urdf_path", type=str, default="example_6dof_manipulator.urdf", 
                        help="URDF file path or name (default: example_6dof_manipulator.urdf)")
    parser.add_argument("--render", action="store_true", help="Enable rendering in PyBullet")
    parser.add_argument("--no-render", action="store_false", dest="render", help="Disable rendering in PyBullet")
    parser.add_argument("--demo", choices=["rrt", "cartesian_two_stage", "rrt_amiga", "cached_trajectory"],
                        default="rrt",
                        help="Which demo to run (default: rrt). 'cartesian_two_stage' and "
                             "'rrt_amiga' both run on the UR5e-on-Amiga scenario, planning with "
                             "the two-stage Cartesian planner and RRT-Connect respectively. "
                             "'cached_trajectory' plays back paths saved by trajectory_cache "
                             "(see --paths-file).")
    parser.add_argument("--paths-file", type=str, default=None,
                        help="cached_trajectory: reachable_paths*.npy file, or a run directory "
                             "containing a reachable_paths_merged_*.npy.")
    parser.add_argument("--index", type=int, default=None,
                        help="cached_trajectory: path index to play first (more can be chosen at "
                             "the prompt afterward).")
    parser.add_argument("--urdf-dir", type=str, default=None,
                        help="Amiga demos: directory holding ur5e/ur5e.urdf and amiga/visual/ "
                             "(e.g. trajectory_cache/urdf). Defaults to this package's urdf/robots.")
    parser.add_argument("--base-yaw-deg", type=float, default=45.0,
                        help="cached_trajectory: robot base yaw in degrees; must match what the "
                             "paths were generated with (default: 45).")
    parser.add_argument("--ee-link", type=str, default="gripper_link",
                        help="cached_trajectory: end-effector link name (default: gripper_link).")
    parser.add_argument("--delay", type=float, default=0.01,
                        help="cached_trajectory: seconds between configs during playback (default: 0.01).")

    parser.set_defaults(render=True)  # Default to True
    args = parser.parse_args()

    if args.demo == "cached_trajectory":
        if args.paths_file is None:
            parser.error("--demo cached_trajectory requires --paths-file")
        view_cached_trajectories(args.paths_file, urdf_dir=args.urdf_dir,
                                 robot_base_yaw=np.deg2rad(args.base_yaw_deg), ee_link_name=args.ee_link,
                                 index=args.index, delay=args.delay)
        raise SystemExit
    elif args.demo == "cartesian_two_stage":
        two_stage_cartesian_demo(renders=args.render)
        raise SystemExit
    elif args.demo == "rrt_amiga":
        rrt_amiga_demo(renders=args.render)
        raise SystemExit

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_urdf_dir = os.path.join(script_dir, 'urdf', 'robots')
    default_urdf_file = os.path.join(default_urdf_dir, "example_6dof_manipulator.urdf")
    
    robot_urdf_path = get_urdf_path(args.urdf_path, default_urdf_dir, default_urdf_file)

    robot_home_pos = None

    print(f"\nLoading robot from: {robot_urdf_path}\n")

    view_robot = ViewRobot(robot_urdf_path=robot_urdf_path,
                           renders=args.render,
                           robot_home_pos=robot_home_pos,
                           ik_tol=0.1,
                           ee_link_name=None)
    
    # view_robot.test_resolved_rate_motion_control()
    # view_robot.main() 
    view_robot.rrt_path_test()
    # view_robot.test_collisions()
