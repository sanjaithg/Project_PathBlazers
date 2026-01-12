#!/home/hillman/ROS2_NEW/pathblazers/.venv/bin/python3
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bots_path = "/home/hillman/ROS2_NEW/pathblazers/src/bots"
    
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
