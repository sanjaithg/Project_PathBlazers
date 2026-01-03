import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
def generate_launch_description():

    # Use package-relative paths for portability
    pkg_bots = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Parent of 'launch' folder
    ros_gz_sim_path = get_package_share_directory('ros_gz_sim')

    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='True',
        description='Flag to enable use_sim_time'
    )


    # Append gazebo models path to env var (use home directory relative path)
    home_dir = os.path.expanduser("~")
    gazebo_models_path = os.path.join(home_dir, "ROS2_NEW/gazebo_models")
    os.environ["GZ_SIM_RESOURCE_PATH"] = os.environ.get("GZ_SIM_RESOURCE_PATH", "") + os.pathsep + gazebo_models_path

    # Launch arguments (simple defaults, no substitution)
    rviz_launch_arg = DeclareLaunchArgument('rviz', default_value='true', description='Open RViz.')
    world_arg = DeclareLaunchArgument('world', default_value=os.path.join(pkg_bots, 'world', 'world.sdf'), description='Absolute path to Gazebo world file')
    model_arg = DeclareLaunchArgument('model', default_value=os.path.join(pkg_bots, 'urdf', 'mogi_bot_mecanum.urdf'), description='Absolute path to URDF file')

    # Include Gazebo world launch file with absolute path argument
    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_bots, 'launch', 'my_launch_file.py')),
        launch_arguments={'world': LaunchConfiguration('world')}.items()
    )
    
    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(pkg_bots, 'rviz', 'rviz.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    # Spawn robot node using absolute URDF path via 'robot_description' topic
    spawn_urdf_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-name', 'my_robot', '-topic', 'robot_description', '-x', '1', '-y', '1', '-z', '0.5', '-Y', '0'],
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    # Robot state publisher launching with absolute URDF path via xacro command
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': Command(['xacro ', LaunchConfiguration('model')]),
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')]
    )
     
    # Node to bridge /cmd_vel and /odom
    gz_bridge_node = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model",
            #"/tf@tf2_msgs/msg/TFMessage@gz.msgs.Pose_V",
            #"/camera/image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/imu@sensor_msgs/msg/Imu@gz.msgs.IMU",  # ENABLED for yaw correction
            # "/navsat@sensor_msgs/msg/NavSatFix@gz.msgs.NavSat",
            "/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan",
            # "/scan/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
            "/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
            # Ground Truth Pose (GZ to ROS2 only - use '[' not '@')
            "/model/my_robot/pose@geometry_msgs/msg/Pose[gz.msgs.Pose",
        ],
        output="screen",
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
    )

    # Node to bridge camera image with image_transport and compressed_image_transport
    gz_image_bridge_node = Node(
        package="ros_gz_image",
        executable="image_bridge",
        arguments=[
            "/camera/image",
        ],
        output="screen",
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time'),
             'camera.image.compressed.jpeg_quality': 75},
        ],
    )

    # Relay node to republish /camera/camera_info to /camera/image/camera_info
    relay_camera_info_node = Node(
        package='topic_tools',
        executable='relay',
        name='relay_camera_info',
        output='screen',
        arguments=['camera/camera_info', 'camera/image/camera_info'],
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
    )

    trajectory_node = Node(
        package='mogi_trajectory_server',
        executable='mogi_trajectory_server',
        name='mogi_trajectory_server',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            os.path.join(pkg_bots, 'config', 'ekf.yaml'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
             ]
    )

    trajectory_odom_topic_node = Node(
        package='mogi_trajectory_server',
        executable='mogi_trajectory_server_topic_based',
        name='mogi_trajectory_server_odom_topic',
        parameters=[{'trajectory_topic': 'trajectory_raw'},
                    {'odometry_topic': 'odom'}]
    )

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(rviz_launch_arg)
    # launchDescriptionObject.add_action(rviz_config_arg)
    launchDescriptionObject.add_action(world_arg)
    launchDescriptionObject.add_action(model_arg)
    # launchDescriptionObject.add_action(x_arg)
    # launchDescriptionObject.add_action(y_arg)
    # launchDescriptionObject.add_action(yaw_arg)
    launchDescriptionObject.add_action(sim_time_arg)
    launchDescriptionObject.add_action(world_launch)
    launchDescriptionObject.add_action(rviz_node)
    launchDescriptionObject.add_action(spawn_urdf_node)
    launchDescriptionObject.add_action(gz_bridge_node)
    launchDescriptionObject.add_action(gz_image_bridge_node)
    launchDescriptionObject.add_action(relay_camera_info_node)
    launchDescriptionObject.add_action(robot_state_publisher_node)
    # launchDescriptionObject.add_action(trajectory_node)
    launchDescriptionObject.add_action(ekf_node)
    # launchDescriptionObject.add_action(trajectory_odom_topic_node)

    return launchDescriptionObject
