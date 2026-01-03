try:
    from ros_gz_interfaces.srv import GetEntityState
    print("GetEntityState found")
except ImportError:
    print("GetEntityState NOT found")

try:
    from gazebo_msgs.srv import GetEntityState as GZGetEntityState
    print("gazebo_msgs GetEntityState found")
except ImportError:
    print("gazebo_msgs GetEntityState NOT found")
