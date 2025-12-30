#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading

class OdomVisualizer(Node):
    def __init__(self):
        super().__init__('odom_visualizer')
        self.x_data = []
        self.y_data = []

        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)

        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'b-', label='Path')
        self.ax.set_xlabel('X position (m)')
        self.ax.set_ylabel('Y position (m)')
        self.ax.set_title('Odometry Path Visualization')
        self.ax.legend()
        self.ax.grid()
        self.ax.axis('equal')

        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100)

    def odom_callback(self, msg):
        self.x_data.append(msg.pose.pose.position.x)
        self.y_data.append(msg.pose.pose.position.y)
        # optional: limit buffer size
        if len(self.x_data) > 1000:
            self.x_data.pop(0)
            self.y_data.pop(0)

    def update_plot(self, frame):
        self.line.set_data(self.x_data, self.y_data)
        if self.x_data and self.y_data:
            self.ax.set_xlim(min(self.x_data) - 1, max(self.x_data) + 1)
            self.ax.set_ylim(min(self.y_data) - 1, max(self.y_data) + 1)
        return self.line,

def ros_spin(node):
    rclpy.spin(node)

def main(args=None):
    rclpy.init(args=args)
    node = OdomVisualizer()

    # Start ROS spin in a separate thread to handle subscriptions
    spin_thread = threading.Thread(target=ros_spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        plt.show()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

