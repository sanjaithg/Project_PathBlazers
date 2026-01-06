
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import  LaunchConfiguration, PathJoinSubstitution, TextSubstitution


def generate_launch_description():

    pkg_bots = get_package_share_directory('bots')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    world_arg = DeclareLaunchArgument(
        'world', default_value='world.sdf',
        description='World file name relative to the bots/world directory'
    )

    resource_search_paths = [
        os.environ.get("GZ_SIM_RESOURCE_PATH", ""),
        pkg_bots,
        os.path.join(pkg_bots, "world"),
        os.path.join(pkg_bots, "world", "models"),
        os.path.join(pkg_bots, "meshes"),
        "/usr/share/gz/gz-sim8/models",
        os.path.expanduser("~/.gz/fuel"),
    ]
    os.environ["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(
        [path for path in resource_search_paths if path]
    )
    os.environ["GZ_IP"] = "127.0.0.1"
    os.environ["GZ_IP"] = "127.0.0.1"


    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py'),
        ),
        launch_arguments={'gz_args': [PathJoinSubstitution([
            pkg_bots,
            'world',
            LaunchConfiguration('world')
        ]),
        #TextSubstitution(text=' -r -v -v1 --render-engine ogre --render-engine-gui-api-backend opengl')],
        TextSubstitution(text=' -r -v -v1')],
        'on_exit_shutdown': 'true'}.items()
    )

    launchDescriptionObject = LaunchDescription()

    launchDescriptionObject.add_action(world_arg)
    launchDescriptionObject.add_action(gazebo_launch)

    return launchDescriptionObject
