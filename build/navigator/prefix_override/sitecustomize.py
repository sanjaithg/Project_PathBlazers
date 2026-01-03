import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/dark/ros2_ws/src2/Project_PathBlazers/install/navigator'
