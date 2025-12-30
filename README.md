# Project PathBlazers

A ROS2 project for robot navigation and reinforcement learning using TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm.

## Project Overview

This repository contains ROS2 packages for:
- **bots**: Robot simulation and TD3 reinforcement learning training
- **navigator**: Robot navigation and odometry visualization tools

## Prerequisites

- ROS2 Jazzy (or compatible version)
- Python 3
- Gazebo simulation environment
- Required Python packages (see requirements.txt)

## Installation Instructions

### Clone the Repository into Your ROS2 Workspace

To clone this repository into your ROS2 workspace:

**Option 1: Clone into a custom `src2` directory:**
```bash
# Create the src2 directory if it doesn't exist
mkdir -p ~/ros2_ws/src2

# Navigate to your ROS2 workspace
cd ~/ros2_ws/src2

# Clone the repository
git clone https://github.com/sanjaithg/Project_PathBlazers.git

# Navigate into the cloned directory
cd Project_PathBlazers
```

**Option 2: Clone into the standard `src` directory:**
```bash
# Navigate to your ROS2 workspace
cd ~/ros2_ws/src

# Clone the repository
git clone https://github.com/sanjaithg/Project_PathBlazers.git

# Navigate into the cloned directory
cd Project_PathBlazers
```

> **Note:** The examples below assume you cloned into `src2`. If you used `src`, adjust the paths accordingly.

### Install Dependencies

1. **Install Python dependencies:**
   ```bash
   # Navigate to the src directory within the cloned repository
   # If you cloned into src2:
   cd ~/ros2_ws/src2/Project_PathBlazers/src
   # If you cloned into src:
   # cd ~/ros2_ws/src/Project_PathBlazers/src
   
   # Install dependencies from requirements.txt
   pip install -r requirements.txt
   ```

2. **Install additional dependencies for odometry visualization:**
   ```bash
   pip install matplotlib
   ```

### Build the Workspace

```bash
# Navigate back to your ROS2 workspace root
cd ~/ros2_ws

# Source ROS2
source /opt/ros/jazzy/setup.bash

# Build the packages
colcon build --packages-select bots navigator

# Source the workspace
source install/setup.bash
```

## Usage

### Odometry Visualization

Run the odometry visualizer to track and visualize robot paths:

```bash
# Make sure ROS2 is sourced
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

# Run the visualizer (adjust path based on where you cloned)
# If you cloned into src2:
python3 ~/ros2_ws/src2/Project_PathBlazers/odom_visualizer_ros2_jazzy.py
# If you cloned into src:
# python3 ~/ros2_ws/src/Project_PathBlazers/odom_visualizer_ros2_jazzy.py
```

### Launch Robot Simulation

The project includes several launch files for different purposes:

```bash
# Spawn robot in Gazebo
ros2 launch bots spawn_robot.launch.py

# Train TD3 reinforcement learning model
ros2 launch bots train_td3.launch.py
```

## Package Structure

```
Project_PathBlazers/
├── odom_visualizer_ros2_jazzy.py  # Standalone odometry visualization tool
└── src/
    ├── bots/                       # Robot simulation and RL training
    │   ├── bots/
    │   │   └── td3_rl/            # TD3 reinforcement learning implementation
    │   ├── launch/                # Launch files
    │   └── package.xml
    ├── navigator/                  # Navigation package
    │   └── package.xml
    ├── lerobots/                   # Robot configurations
    ├── mogi_trajectory_server/     # Trajectory server
    └── requirements.txt            # Python dependencies
```

## Packages

### bots
Contains the robot URDF models, Gazebo simulation setup, and TD3 reinforcement learning implementation for autonomous navigation training.

### navigator
Navigation utilities and tools for robot path planning and odometry handling.

## Contributing

This is a research and development project for robot navigation using reinforcement learning.

## License

TODO: License declaration

## Maintainer

- hillman (ed23b055@smail.iitm.ac.in)
