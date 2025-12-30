from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        Node(
            package='navigator',
            executable='front',
            name='front_wheels',
            output='screen'
        ),

        Node(
            package='navigator',
            executable='rear',
            name='rear_wheels',
            output='screen'
        ),

        Node(
            package='navigator',
            executable='inv_kin',
            name='inv_kin',
            output='screen'
     	),
    ])

