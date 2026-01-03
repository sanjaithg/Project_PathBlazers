#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

# ✅ Gazebo Harmonic uses gz.transport15
import gz.transport15 as gz_transport
from gz.msgs.pose_v_pb2 import Pose_V


class GazeboTruthPoseOnly(Node):
    def __init__(self):
        super().__init__('gz_truth_pose_only_to_odom')

        self.robot_name = "my_robot"   # 🔧 change this to match your entity name in Gazebo
        self.latest_pose = None

        # TF broadcaster
        self.tf_br = TransformBroadcaster(self)

        # Subscribe to Gazebo ground-truth poses
        self.gz_sub = gz_transport.Subscriber(
            "/world/default/dynamic_pose/info", Pose_V, self.gazebo_pose_callback
        )

        # Subscribe to the existing odom (so we keep twist)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )

        # Republish updated odom (same topic)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        self.get_logger().info(
            f"Using Gazebo Harmonic ground-truth pose for '{self.robot_name}', keeping odom twist as-is."
        )

    def gazebo_pose_callback(self, msg):
        """Store the latest Gazebo pose for our robot."""
        for pose in msg.pose:
            if pose.name == self.robot_name:
                self.latest_pose = pose
                break

    def odom_callback(self, odom_msg):
        """Replace odom pose with Gazebo truth, keep twist unchanged."""
        if self.latest_pose is None:
            return

        # Replace pose (ground-truth)
        odom_msg.pose.pose.position.x = self.latest_pose.position.x
        odom_msg.pose.pose.position.y = self.latest_pose.position.y
        odom_msg.pose.pose.position.z = self.latest_pose.position.z

        odom_msg.pose.pose.orientation.x = self.latest_pose.orientation.x
        odom_msg.pose.pose.orientation.y = self.latest_pose.orientation.y
        odom_msg.pose.pose.orientation.z = self.latest_pose.orientation.z
        odom_msg.pose.pose.orientation.w = self.latest_pose.orientation.w

        # Keep twist from incoming message
        # Update timestamp
        odom_msg.header.stamp = self.get_clock().now().to_msg()

        # Publish modified message
        self.odom_pub.publish(odom_msg)

        # Broadcast TF (odom → base_link)
        t = TransformStamped()
        t.header.stamp = odom_msg.header.stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = self.latest_pose.position.x
        t.transform.translation.y = self.latest_pose.position.y
        t.transform.translation.z = self.latest_pose.position.z
        t.transform.rotation = odom_msg.pose.pose.orientation
        self.tf_br.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = GazeboTruthPoseOnly()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

