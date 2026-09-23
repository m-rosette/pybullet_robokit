# pybullet_robokit

A PyBullet-based toolkit for simulating and controlling robot arms — built around inverse kinematics, motion planning, and collision checking, with utilities for voxelizing 3D geometry (e.g. tree/plant canopies) for obstacle representation.

## What it does

- **`PybUtils`** ([pyb_utils.py](pybullet_robokit/pyb_utils.py)) — Spins up a PyBullet client (GUI or headless `DIRECT` mode) with sane defaults (gravity, timestep, camera).
- **`LoadObjects`** ([load_objects.py](pybullet_robokit/load_objects.py)) — Loads URDFs (ground plane, obstacles, targets) into the simulation.
- **`LoadRobot`** ([load_robot.py](pybullet_robokit/load_robot.py)) — Loads a robot arm URDF and exposes joint control, forward/inverse kinematics, Jacobians, manipulability, and self/environment collision checking (including auto-disabling of collisions between adjacent/spherical joint groups).
- **`KinematicChainMotionPlanner`** ([motion_planners.py](pybullet_robokit/motion_planners.py)) — Joint-space and Cartesian-space trajectory generation, a two-stage Cartesian planner (world X/Z then Y, with an optional per-waypoint collision-avoidance variant), RRT-Connect planning (via `pybullet_planning`, with joint-weighted distance and a shortcutting/smoothing pass to curb erratic joint swings), and resolved-rate motion control with manipulability-maximizing nullspace biasing.
- **`voxel_gen.py`** ([voxel_gen.py](pybullet_robokit/voxel_gen.py)) — Point cloud filtering/downsampling and mesh voxelization (via Open3D), including a parametric parallelepiped generator for representing objects like V-trellis canopies as voxel obstacles.
- **`ViewRobot`** ([view_robot.py](pybullet_robokit/view_robot.py)) — Example/demo entry point. `ViewRobot` loads an example 6-DOF arm next to a V-trellis structure for Cartesian path, RRT path, resolved-rate control, or collision-check demos; `--demo cartesian_two_stage`/`rrt_amiga` instead run a UR5e mounted on an Amiga mobile base, with the Amiga's chassis and hardware modeled as collision obstacles.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

```bash
source .venv/bin/activate
python -m pybullet_robokit.view_robot --render
```

Use `-u/--urdf_path` to point at a different robot URDF (relative to `pybullet_robokit/urdf/robots/` or an absolute path), `--no-render` to run headless, and `--demo {rrt,cartesian_two_stage,rrt_amiga}` to pick which planning demo to run (default: `rrt`).
