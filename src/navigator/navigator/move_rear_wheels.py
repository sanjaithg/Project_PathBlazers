#!/usr/bin/env python3

import rclpy
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
import serial
import time
import numpy as np

class HwInterface(Node):
    def __init__(self):
        super().__init__('hw_interface_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('wheel_radius', 37),
                ('robot_width', 475),
                ('max_speed', 35),
                ('min_speed', 0.1),
                ('separation_bet_wheels', 50),
                ('serial_port', '/dev/ttyREAR'),
                ('baud_rate', 9600)
            ]
        )

        # ROS2 Parameters
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.max_speed = self.get_parameter('max_speed').value
        self.serial_port = self.get_parameter('serial_port').value
        self.baud_rate = self.get_parameter('baud_rate').value

        # Serial communication setup
        self.startMarker = 60  # '<'
        self.endMarker = 62    # '>'
        self.ser = serial.Serial(self.serial_port, self.baud_rate)
        self.wait_for_arduino()

        # ROS2 Publishers/Subscribers
        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            qos
        )
        
        self.rear_left_pub = self.create_publisher(Int32, '/rear_left_ticks', qos)
        self.rear_right_pub = self.create_publisher(Int32, '/rear_right_ticks', qos)

        # Control variables
        self.front_left_w = 0.0
        self.front_right_w = 0.0
        self.rear_left_w = 0.0
        self.rear_right_w = 0.0

        # Main control timer
        self.control_timer = self.create_timer(0.25, self.control_loop)

    def wait_for_arduino(self):
        """Wait for Arduino initialization sequence"""
        self.get_logger().info('Waiting for Arduino to initialize...')
        msg = ""
        while msg.find("Arduino is ready") == -1 and rclpy.ok():
            while self.ser.in_waiting == 0 and rclpy.ok():
                time.sleep(0.01)
            msg = self.recv_from_arduino()
        self.get_logger().info('Arduino connection established')

    def recv_from_arduino(self):
        """Receive and parse serial data from Arduino"""
        received_data = ""
        received_char = b'!'
        
        # Wait for start marker
        while ord(received_char) != self.startMarker and rclpy.ok():
            received_char = self.ser.read(1)
        
        # Read until end marker
        while ord(received_char) != self.endMarker and rclpy.ok():
            if ord(received_char) != self.startMarker:
                received_data += received_char.decode('utf-8')
            received_char = self.ser.read(1)
        
        return received_data

    
    def cmd_vel_callback(self, msg):
        """Convert Twist message to wheel velocities"""
        a = 0.076  # Wheel radius
        l = 0.27   # Length between front and rear wheels
        d = 0.215  # Track width

        # Kinematic calculations
        self.front_left_w = (1/a) * (msg.linear.x - msg.linear.y - (l + d) * msg.angular.z)
        self.front_right_w = (1/a) * (msg.linear.x + msg.linear.y + (l + d) * msg.angular.z)
        self.rear_left_w = (1/a) * (msg.linear.x + msg.linear.y - (l + d) * msg.angular.z)
        self.rear_right_w = (1/a) * (msg.linear.x - msg.linear.y + (l + d) * msg.angular.z)
    def control_loop(self):
        """Main control loop for sending commands and receiving feedback"""
        # Send wheel commands
        output = f"<{int(self.rear_left_w * self.max_speed)*0.5},{int(self.rear_right_w * self.max_speed)*0.5}>"
        print(output)
        self.ser.reset_output_buffer()
        self.ser.write(output.encode('utf-8'))

        # Receive encoder data
        self.ser.reset_input_buffer()

        data = self.recv_from_arduino()
        ticks = [int(float(x)) for x in data.split(',')]
        print("tick data: ", ticks)
        # Publish encoder ticks
        self.rear_left_pub.publish(Int32(data=ticks[0]))
        self.rear_right_pub.publish(Int32(data=ticks[1]))
        # try:
        #     if self.ser.in_waiting > 0:
        #         data = self.recv_from_arduino()
        #         ticks = [int(float(x)) for x in data.split(',')]
        #         print("tick data: ", ticks)
        #         # Publish encoder ticks
        #         self.rear_left_pub.publish(Int64(data=ticks[0]))
        #         self.rear_right_pub.publish(Int64(data=ticks[1]))
        # except serial.SerialException as e:
        #     self.get_logger().error(f"Serial communication error: {str(e)}")
        #     print("nothing came")
        #     self.safe_shutdown()

    def safe_shutdown(self):
        """Graceful shutdown procedure"""
        self.get_logger().info('Performing safety shutdown')
        self.ser.write("<0,0>".encode('utf-8'))
        time.sleep(1)
        self.ser.close()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    try:
        interface = HwInterface()
        rclpy.spin(interface)
    except KeyboardInterrupt:
        interface.get_logger().info('Node terminated by user')
    except Exception as e:
        interface.get_logger().fatal(f'Fatal error: {str(e)}')
    finally:
        interface.safe_shutdown()
        interface.destroy_node()

if __name__ == '__main__':
    main()

