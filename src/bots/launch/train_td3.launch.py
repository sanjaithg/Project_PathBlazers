#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Use relative path for portability
    bots_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', description='Launch Gazebo with GUI'),
        DeclareLaunchArgument('rviz', default_value='true', description='Launch RViz'),
        DeclareLaunchArgument('world', default_value=os.path.join(bots_path, 'world', 'world_simplified.sdf'), description='Gazebo world file'),
        
        # Start Gazebo and spawn robot
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([bots_path, 'launch', 'spawn_robot.launch.py'])),
            launch_arguments={
                'gui': LaunchConfiguration('gui'), 
                'rviz': LaunchConfiguration('rviz'),
                'world': LaunchConfiguration('world')
            }.items(),
        ),
    ])
