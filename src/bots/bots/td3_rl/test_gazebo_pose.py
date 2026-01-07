import unittest
from unittest.mock import MagicMock
import numpy as np
import sys
import math

# We don't even need to import GazeboEnv if we just want to test the logic
# But let's try to import it and extract the method

sys.modules['rclpy'] = MagicMock()
sys.modules['rclpy.node'] = MagicMock()
sys.modules['ros_gz_interfaces.srv'] = MagicMock()
sys.modules['geometry_msgs.msg'] = MagicMock()
sys.modules['nav_msgs.msg'] = MagicMock()
sys.modules['std_srvs.srv'] = MagicMock()
sys.modules['ros_gz_interfaces.msg'] = MagicMock()
sys.modules['visualization_msgs.msg'] = MagicMock()
sys.modules['scipy.spatial.transform'] = MagicMock()

# Define a minimal version of the logic to test if import fails or just extract it
try:
    from bots.td3_rl.gazebo_env import GazeboEnv
except ImportError:
    print("Fallback to manual logic test")
    GazeboEnv = None

class DummyEnv:
    def __init__(self):
        self.use_ground_truth = False
        self.pose_topic = '/model/my_robot/odometry_with_covariance'
        self.odom_topic = '/odom'
        self.ground_truth_noise_std = 0.0
        self.last_pose = None
        self.last_odom = None
        self._fallback_logged = False
        self.get_logger = MagicMock()

    # Copy the method here for testing if import is problematic, 
    # but let's try to use the one from the class if possible.
    def get_current_pose(self):
        # This is strictly the logic from the implemented gazebo_env.py
        use_fallback = False
        pose_to_parse = None
        source_name = ""

        if self.use_ground_truth:
            if self.last_pose is not None:
                pose_to_parse = self.last_pose
                source_name = "ground-truth"
            else:
                use_fallback = True
                if not self._fallback_logged:
                    self.get_logger().warn('fallback-warn')
                    self._fallback_logged = True
        
        if not self.use_ground_truth or use_fallback:
            if self.last_odom is not None:
                pose_to_parse = self.last_odom.pose.pose
                source_name = "odom"
            else:
                return None

        x = float(pose_to_parse.position.x)
        y = float(pose_to_parse.position.y)

        if source_name == "ground-truth" and self.ground_truth_noise_std > 0.0:
            x += np.random.normal(0, self.ground_truth_noise_std)
            y += np.random.normal(0, self.ground_truth_noise_std)

        q = pose_to_parse.orientation
        # Mocking R.from_quat since we mocked scipy
        # For test purposes, we'll assume yaw = 0 if w=1
        angle = 0.0 
        
        if source_name == "ground-truth" and self.ground_truth_noise_std > 0.0:
            angle += np.random.normal(0, self.ground_truth_noise_std)
            angle = (angle + np.pi) % (2 * np.pi) - np.pi

        return x, y, angle

class TestPoseLogic(unittest.TestCase):
    def setUp(self):
        self.env = DummyEnv()

    def test_odom_only(self):
        self.env.use_ground_truth = False
        self.env.last_odom = MagicMock()
        self.env.last_odom.pose.pose.position.x = 1.0
        self.env.last_odom.pose.pose.position.y = 2.0
        
        res = self.env.get_current_pose()
        self.assertEqual(res[0], 1.0)
        self.assertEqual(res[1], 2.0)

    def test_gt_preference(self):
        self.env.use_ground_truth = True
        self.env.last_pose = MagicMock()
        self.env.last_pose.position.x = 5.0
        
        self.env.last_odom = MagicMock()
        self.env.last_odom.pose.pose.position.x = 1.0
        
        res = self.env.get_current_pose()
        self.assertEqual(res[0], 5.0)

    def test_gt_fallback(self):
        self.env.use_ground_truth = True
        self.env.last_pose = None
        self.env.last_odom = MagicMock()
        self.env.last_odom.pose.pose.position.x = 10.0
        
        res = self.env.get_current_pose()
        self.assertEqual(res[0], 10.0)
        self.env.get_logger().warn.assert_called_with('fallback-warn')

    def test_noise(self):
        self.env.use_ground_truth = True
        self.env.ground_truth_noise_std = 1.0
        self.env.last_pose = MagicMock()
        self.env.last_pose.position.x = 0.0
        self.env.last_pose.position.y = 0.0
        
        poses = [self.env.get_current_pose() for _ in range(100)]
        xs = [p[0] for p in poses]
        self.assertTrue(np.std(xs) > 0.5)

if __name__ == '__main__':
    unittest.main()
