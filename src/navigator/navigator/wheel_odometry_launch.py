import os
import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    urdf_file = "/home/ibot/ros2_ws/src/navigator/navigator/box_bot.urdf"

    return launch.LaunchDescription([
        # Start Wheel Odometry Node
        launch_ros.actions.Node(
            package='your_package_name',
            executable='wheel_odometry',
            output='screen'
        ),

        # Run Robot State Publisher
        launch_ros.actions.Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': open(urdf_file).read()}]
        ),

        # Run TF broadcaster for odom → base_link
        launch_ros.actions.Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link']
        ),

        # Run TF broadcaster for world → odom
        launch_ros.actions.Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'odom']
        ),

        # Open RViz
        launch_ros.actions.Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', "/home/ibot/ros2_ws/src/navigator/rviz/box_bot.rviz"],
            output='screen'
        ),
    ])
