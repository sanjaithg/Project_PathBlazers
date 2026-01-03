from pathlib import Path
import launch
import launch_ros.actions

def generate_launch_description():
    pkg_root = Path(__file__).resolve().parent
    urdf_file = pkg_root / "box_bot.urdf"
    rviz_config = pkg_root.parent / "rviz" / "box_bot.rviz"

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
            arguments=['-d', str(rviz_config)],
            output='screen'
        ),
    ])
