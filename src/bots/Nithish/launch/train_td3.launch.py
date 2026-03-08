#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    from ament_index_python.packages import get_package_share_directory
    bots_path = get_package_share_directory('bots')
    
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', description='Launch Gazebo with GUI'),
        DeclareLaunchArgument('rviz', default_value='true', description='Launch RViz'),
        
        # Start Gazebo and spawn robot
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([bots_path, 'launch', 'spawn_robot.launch.py'])),
            launch_arguments={
                'gui': LaunchConfiguration('gui'), 
                'rviz': LaunchConfiguration('rviz')
            }.items(),
        ),

        # # Start the TD3 training script with parameters loaded from YAML
        # Node(
        #     package='bots',
        #     executable='train_td3',
        #     name='td3_trainer',
        #     output='screen',	
        #     parameters=[PathJoinSubstitution([bots_path, 'config', 'td3_config.yaml'])],
        # ),
    ])
