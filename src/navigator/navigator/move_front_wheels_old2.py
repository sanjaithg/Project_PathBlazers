import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int64
import serial
import time
import numpy as np

class HWInterfaceNode(Node):
    def __init__(self):
        super().__init__('hw_interface_node')
        
        self.ser = serial.Serial("/dev/ttyUSB0", 9600)
        self.start_marker = 60  # Unicode code for '<'
        self.end_marker = 62    # Unicode code for '>'
        self.max_speed = 35
        self.min_speed = 0.1
        
        self.front_left_w, self.front_right_w = 0.0, 0.0
        self.rear_left_w, self.rear_right_w = 0.0, 0.0
        
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.convert_vel_cmd,
            10)
        
        self.pub_front_left_ticks = self.create_publisher(Int64, '/front_left_ticks', 10)
        self.pub_front_right_ticks = self.create_publisher(Int64, '/front_right_ticks', 10)
        
        self.declare_parameter('wheel_radius', 37)
        self.declare_parameter('robot_width', 475)
        
        if not self.wait_for_arduino():
            self.get_logger().error("Arduino not ready. Exiting.")
            return
        
        self.timer = self.create_timer(0.25, self.main_loop)
        
    def wait_for_arduino(self):
        self.get_logger().info("Waiting for Arduino to be ready...")
        msg = ""
        while "Arduino is ready" not in msg:
            if not self.ser.in_waiting:
                time.sleep(0.1)
                continue
            msg = self.recv_from_arduino()
        return True

    def recv_from_arduino(self):
        received_data = ""
        received_char = "!"
        while ord(received_char) != self.start_marker:
            received_char = self.ser.read()
        while ord(received_char) != self.end_marker:
            if ord(received_char) != self.start_marker:
                received_data += received_char.decode('utf-8')
            received_char = self.ser.read()
        return received_data

    def convert_vel_cmd(self, msg):
        a, l, d = 0.076, 0.27, 0.215
        x_vel, y_vel, w_vel = msg.linear.x, msg.angular.z, msg.linear.y
        
        self.front_left_w = (1 / a) * (x_vel - y_vel - (l + d) * w_vel)
        self.front_right_w = (1 / a) * (x_vel + y_vel + (l + d) * w_vel)
        self.rear_left_w = (1 / a) * (x_vel + y_vel - (l + d) * w_vel)
        self.rear_right_w = (1 / a) * (x_vel - y_vel + (l + d) * w_vel)

    def main_loop(self):
        output_string = f"<{int(self.front_left_w * self.max_speed)},{int(self.front_right_w * self.max_speed)}>"
        self.ser.reset_output_buffer()
        self.ser.write(output_string.encode('utf-8'))
        
        self.ser.reset_input_buffer()
        while not self.ser.in_waiting:
            time.sleep(0.01)
        
        received_data = self.recv_from_arduino()
        try:
            ticks = [int(float(x)) for x in received_data.split(',')]
            self.pub_front_left_ticks.publish(Int64(data=ticks[0]))
            self.pub_front_right_ticks.publish(Int64(data=ticks[1]))
        except ValueError:
            self.get_logger().warn("Received invalid data from Arduino: " + received_data)

    def turn_off(self):
        self.get_logger().info("Shutting down hw_interface_node...")
        self.ser.write("<0,0>".encode('utf-8'))
        time.sleep(3)
        self.ser.close()
        

def main():
    rclpy.init()
    node = HWInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.turn_off()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()