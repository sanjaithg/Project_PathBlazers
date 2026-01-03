import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from nav_msgs.msg import Odometry
import math
import tf2_ros
from geometry_msgs.msg import TransformStamped

class WheelOdometry(Node):
    def __init__(self):
        super().__init__('wheel_odometry')

        # Robot parameters
        self.ticks_per_rev = 350  # Encoder resolution
        self.wheel_radius = 0.075  # Wheel radius (meters)
        self.wheel_base = 0.53     # Distance between wheels

        # Store previous encoder ticks
        self.prev_ticks = {'fl': 0, 'fr': 0, 'rl': 0, 'rr': 0}
        self.current_ticks = {'fl': 0, 'fr': 0, 'rl': 0, 'rr': 0}

        # Subscribe to wheel encoders
        self.create_subscription(Int32, '/front_left_ticks', self.fl_callback, 10)
        self.create_subscription(Int32, '/front_right_ticks', self.fr_callback, 10)
        self.create_subscription(Int32, '/rear_left_ticks', self.rl_callback, 10)
        self.create_subscription(Int32, '/rear_right_ticks', self.rr_callback, 10)

        # Odometry publisher
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Robot pose
        self.x, self.y, self.theta = 0.0, 0.0, 0.0

        # Timer to compute odometry every 50ms (20Hz)
        self.create_timer(0.05, self.compute_odometry)

    def fl_callback(self, msg):
        self.current_ticks['fl'] = msg.data

    def fr_callback(self, msg):
        self.current_ticks['fr'] = msg.data

    def rl_callback(self, msg):
        self.current_ticks['rl'] = msg.data

    def rr_callback(self, msg):
        self.current_ticks['rr'] = msg.data

    def compute_odometry(self):
        dt = 0.05  # 50ms update rate

        # Compute tick difference
        delta_ticks = {
            'fl': self.current_ticks['fl'] - self.prev_ticks['fl'],
            'fr': self.current_ticks['fr'] - self.prev_ticks['fr'],
            'rl': self.current_ticks['rl'] - self.prev_ticks['rl'],
            'rr': self.current_ticks['rr'] - self.prev_ticks['rr']
        }

        # Convert ticks to distance
        d = {k: (delta_ticks[k] / self.ticks_per_rev) * 2 * math.pi * self.wheel_radius for k in delta_ticks}

        # Compute displacement
        dy = (d['fl'] - d['fr'] + d['rl'] - d['rr']) / 4.0   
        dx = (-d['fl'] - d['fr'] + d['rl'] + d['rr']) / 4.0   
        dtheta = (d['fl'] - d['fr'] - d['rl'] + d['rr']) / (4.0 * self.wheel_base)

        # Update pose
        self.x += dx * math.cos(self.theta) - dy * math.sin(self.theta)
        self.y += dx * math.sin(self.theta) + dy * math.cos(self.theta)
        self.theta = (self.theta + dtheta) % (2 * math.pi)

        # Publish odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"
        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y
        odom_msg.pose.pose.orientation.z = math.sin(self.theta / 2.0)
        odom_msg.pose.pose.orientation.w = math.cos(self.theta / 2.0)
        self.odom_pub.publish(odom_msg)

        # Broadcast TF (odom → base_link)
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = "odom"
        tf_msg.child_frame_id = "base_link"
        tf_msg.transform.translation.x = self.x
        tf_msg.transform.translation.y = self.y
        tf_msg.transform.rotation.z = math.sin(self.theta / 2.0)
        tf_msg.transform.rotation.w = math.cos(self.theta / 2.0)
        self.tf_broadcaster.sendTransform(tf_msg)
        self.get_logger().info(f"Published TF for odom")
        # Update previous ticks
        self.prev_ticks = self.current_ticks.copy()

        # Debug output
        print(f"< {self.x:.4f} , {self.y:.4f} , {self.theta:.4f} >")

def main(args=None):
    rclpy.init(args=args)
    node = WheelOdometry()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
