import rospy
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from lerobot.common.robot import BaseRobot


class MobileRobot(BaseRobot):
    def __init__(self):
        super().__init__()
        self.latest_scan = None
        self.latest_image = None
        self.latest_odom = None

        # Publishers
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Subscribers
        rospy.Subscriber("/scan", LaserScan, self._scan_callback)
        rospy.Subscriber("/odom", Odometry, self._odom_callback)
        rospy.Subscriber("/camera/image_raw", Image, self._image_callback)

    # -------------------- STATE SPACE --------------------
    def get_observation(self):
        """Collect state from sensors."""
        obs = {}

        # LIDAR
        obs["lidar"] = self.latest_scan.ranges if self.latest_scan else [0] * 360

        # ODOMETRY
        obs["odom"] = [
            self.latest_odom.pose.pose.position.x,
            self.latest_odom.pose.pose.position.y,
            self.latest_odom.pose.pose.orientation.z,
        ] if self.latest_odom else [0, 0, 0]

        # CAMERA (Optional)
        # You can add image tensor processing later if needed
        return obs

    # -------------------- ACTION SPACE --------------------
    def apply_action(self, action):
        """Apply control actions to robot."""
        twist = Twist()
        twist.linear.x = action[0]   # forward speed
        twist.angular.z = action[1]  # rotation speed
        self.cmd_pub.publish(twist)

    # -------------------- CALLBACKS --------------------
    def _scan_callback(self, msg):
        self.latest_scan = msg

    def _odom_callback(self, msg):
        self.latest_odom = msg

    def _image_callback(self, msg):
        self.latest_image = msg


# -------------------- REGISTER CUSTOM ROBOT --------------------
from lerobot.robots import register_robot
register_robot("mobile_robot", MobileRobot)

