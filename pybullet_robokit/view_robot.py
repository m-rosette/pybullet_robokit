import os
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
        target_config = self.robot.inverse_kinematics(target_pose, pos_tol=self.ik_tol, max_iter=1000, resample=True, num_resample=10)

        # Initialize the motion planner
        motion_planner = KinematicChainMotionPlanner(self.robot)

        # Pass the start and target configurations to the RRT planner
        joint_path = motion_planner.rrt_path(start_config, target_config, rrt_iter=1000, collision_objects=self.object_loader.collision_objects, steps=500)

        # Check if the path is valid
        if joint_path is None:
            print("\nRRT path planning failed.\n")
            return
        else:
            print("\nRRT path planning succeeded.\n")

        print(len(joint_path), "steps in the path")

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
            joint_config = self.robot.inverse_kinematics(position, pos_tol=self.ik_tol, max_iter=1000, resample=False)

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

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View a robot in PyBullet simulation.")
    parser.add_argument("-u", "--urdf_path", type=str, default="example_6dof_manipulator.urdf", 
                        help="URDF file path or name (default: example_6dof_manipulator.urdf)")
    parser.add_argument("--render", action="store_true", help="Enable rendering in PyBullet")
    parser.add_argument("--no-render", action="store_false", dest="render", help="Disable rendering in PyBullet")
    
    parser.set_defaults(render=True)  # Default to True
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_urdf_dir = os.path.join(script_dir, 'urdf', 'robots')
    default_urdf_file = os.path.join(default_urdf_dir, "example_6dof_manipulator.urdf")
    
    robot_urdf_path = get_urdf_path(args.urdf_path, default_urdf_dir, default_urdf_file)
    ee_link_name = 'end_effector' if "best_chain" or 'test_robot' in robot_urdf_path else 'gripper_link'
    # ee_link_name = 'tool0'
    
    robot_home_pos = None

    print(f"\nLoading robot from: {robot_urdf_path}\n")
    
    view_robot = ViewRobot(robot_urdf_path=robot_urdf_path, 
                           renders=args.render,
                           robot_home_pos=robot_home_pos,
                           ik_tol=0.1,
                           ee_link_name=ee_link_name)
    
    # view_robot.test_resolved_rate_motion_control()
    # view_robot.main() 
    view_robot.rrt_path_test()
    # view_robot.test_collisions()