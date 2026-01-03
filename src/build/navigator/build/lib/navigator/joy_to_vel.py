#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile

class JoyTeleop(Node):
    def __init__(self):
        super().__init__('joy_teleop')  # Initialize the Node properly
        qos = QoSProfile(depth=10)

        self.subscription = self.create_subscription(Joy, '/joy', self.joy_callback, qos)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', qos)

    def joy_callback(self, joy_msg):
        # Extract joystick axes and buttons
        linear_x = joy_msg.axes[1] 
        linear_y = joy_msg.axes[3] 
        angular_z = -joy_msg.axes[0]

        # linear_x = joy_msg.axes[6] * 0.25
        # linear_y = joy_msg.axes[3] 
        # angular_z = -joy_msg.axes[7] *0.25

        # actuator_1_back = joy_msg.axes[2]
        # actuator_2_back = joy_msg.axes[3]
        # actuator_1_for = joy_msg.buttons[4]
        # actuator_2_for = joy_msg.buttons[5]

        # Create Twist message
        cmd_vel_msg = Twist()
        # Explicit type conversion to ensure all fields are floats
        cmd_vel_msg.linear.x = float(linear_x)  # Invert if needed
        cmd_vel_msg.linear.y = float(linear_y)
        cmd_vel_msg.angular.z = float(-angular_z)
        # Handling actuators (if needed for special hardware control, adjust logic here)
        # cmd_vel_msg.angular.x = float(actuator_1_for) if actuator_1_back == 1 else float(actuator_1_back)
        # cmd_vel_msg.angular.y = float(actuator_2_for) if actuator_2_back == 1 else float(actuator_2_back)

        # Publish the command
        self.publisher.publish(cmd_vel_msg)


def main(args=None):
    rclpy.init(args=args)
    joy_teleop = JoyTeleop()
    try:
        rclpy.spin(joy_teleop)
    except KeyboardInterrupt:
        pass
    finally:
        joy_teleop.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
