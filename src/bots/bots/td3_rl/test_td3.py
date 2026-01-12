#!/usr/bin/env python3
import os
import threading
import time
import sys
import numpy as np
import rclpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from bots.td3_rl.gazebo_env import GazeboEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        
        self.ln1 = nn.LayerNorm(800)
        self.ln2 = nn.LayerNorm(600)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.ln1(self.layer_1(s)))
        s = F.relu(self.ln2(self.layer_2(s)))
        a = self.tanh(self.layer_3(s))
        return a

class TD3:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(os.path.join(directory, f"{filename}_actor.pth")))

class TD3Tester(Node):
    def __init__(self):
        super().__init__('td3_tester')
        self.get_logger().info("=" * 50)
        self.get_logger().info("Initializing TD3 Tester Node")
        self.get_logger().info("=" * 50)

        # --- PARAMETERS ---
        self.declare_parameter("seed", 0)
        self.declare_parameter("max_ep", 1000)  # Max steps per goal
        self.declare_parameter("file_name", "TD3_Mecanum")
        self.declare_parameter("environment_dim", 20)
        self.declare_parameter("num_goals", 10)  # Number of goals to test
        self.declare_parameter("model_path", "")  # Optional: explicit path to models folder
        self.declare_parameter("use_ground_truth", False)
        self.declare_parameter("pose_topic", "/model/my_robot/odometry_with_covariance")
        self.declare_parameter("odom_topic", "/odometry/filtered")
        self.declare_parameter("ground_truth_noise_std", 0.0)
        
        # Try to find the models directory
        models_path = self._find_models_path()

        self.seed = self.get_parameter("seed").value
        self.max_ep = self.get_parameter("max_ep").value
        self.file_name = self.get_parameter("file_name").value
        self.environment_dim = self.get_parameter("environment_dim").value
        self.num_goals = self.get_parameter("num_goals").value
        self.use_ground_truth = self.get_parameter("use_ground_truth").value
        self.pose_topic = self.get_parameter("pose_topic").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.ground_truth_noise_std = self.get_parameter("ground_truth_noise_std").value
        
        self.get_logger().info(f"Testing {self.num_goals} random goals")
        self.get_logger().info(f"Max steps per goal: {self.max_ep}")
        self.get_logger().info(f"Loading model: {self.file_name}")

        # Create environment
        env = GazeboEnv(
            environment_dim=self.environment_dim,
            use_ground_truth=self.use_ground_truth,
            pose_topic=self.pose_topic,
            odom_topic=self.odom_topic,
            ground_truth_noise_std=self.ground_truth_noise_std
        )
        
        executor = MultiThreadedExecutor()
        executor.add_node(env)
        env_thread = threading.Thread(target=executor.spin, daemon=True)
        env_thread.start()

        state_dim = self.environment_dim + 14  # Match training (20 + 2 + 3 + 3 + 6)
        action_dim = 3

        self.env = env
        time.sleep(5)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.network = TD3(state_dim, action_dim)
        try:
            self.get_logger().info(f"Loading: {self.file_name}_actor.pth from {models_path}")
            self.network.load(self.file_name, models_path)
            self.get_logger().info("Model loaded successfully!")
        except Exception as e:
            self.get_logger().error(f"Could not load model: {e}")
            sys.exit(1)

        # --- METRICS TRACKING ---
        self.goals_reached = 0
        self.collisions = 0
        self.timeouts = 0
        self.current_goal_num = 0
        self.episode_timesteps = 0
        self.goal_times = []  # Time (steps) to reach each goal
        self.goal_start_time = time.time()
        self.test_start_time = time.time()
        self.testing_complete = False

        # Initialize first goal
        self.env.change_goal()
        self.state = self.env.reset()
        self.current_goal_num = 1
        
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Goal {self.current_goal_num}/{self.num_goals}: ({self.env.goal_x:.2f}, {self.env.goal_y:.2f})")
        self.get_logger().info("=" * 50)

        self.timer = self.create_timer(0.01, self._timer_callback)

    def _find_models_path(self):
        """Find the models directory, checking multiple possible locations."""
        # 1. Check if explicit path was provided
        explicit_path = self.get_parameter("model_path").value
        if explicit_path and os.path.isdir(explicit_path):
            self.get_logger().info(f"Using explicit model_path: {explicit_path}")
            return explicit_path
        
        # 2. Source directory (when running from source)
        home_dir = os.path.expanduser("~")
        source_path = os.path.join(home_dir, "ROS2_NEW/pathblazers/src/bots/bots/td3_rl/pytorch_models")
        if os.path.isdir(source_path):
            self.get_logger().info(f"Found models in source: {source_path}")
            return source_path
        
        # 3. Relative to __file__ (fallback)
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytorch_models")
        if os.path.isdir(local_path):
            self.get_logger().info(f"Found models locally: {local_path}")
            return local_path
        
        # 4. Default fallback
        self.get_logger().warn(f"Models directory not found! Searched: {source_path}, {local_path}")
        return source_path  # Return source path as default

    def _timer_callback(self):
        if not rclpy.ok() or self.testing_complete:
            self.timer.cancel()
            return

        action = self.network.get_action(np.array(self.state))
        a_in = [action[0], action[1], action[2]]

        next_state, reward, done, target_reached = self.env.step(a_in)

        # Check max episode length
        if self.episode_timesteps + 1 >= self.max_ep:
            done = True
            target_reached = False

        if done:
            time_taken = time.time() - self.goal_start_time
            steps_taken = self.episode_timesteps
            
            if target_reached:
                self.goals_reached += 1
                self.goal_times.append(steps_taken)
                self.get_logger().info(f"✓ GOAL {self.current_goal_num} REACHED in {steps_taken} steps ({time_taken:.1f}s)")
            elif reward < -100:  # Collision
                self.collisions += 1
                self.get_logger().warn(f"✗ COLLISION on goal {self.current_goal_num} after {steps_taken} steps")
            else:
                self.timeouts += 1
                self.get_logger().warn(f"✗ TIMEOUT on goal {self.current_goal_num} after {steps_taken} steps")

            # Check if all goals tested
            if self.current_goal_num >= self.num_goals:
                self._print_final_summary()
                self.testing_complete = True
                return

            # Move to next goal
            self.current_goal_num += 1
            self.env.change_goal()  # Generate new random goal
            self.state = self.env.reset()
            self.episode_timesteps = 0
            self.goal_start_time = time.time()
            
            self.get_logger().info("-" * 40)
            self.get_logger().info(f"Goal {self.current_goal_num}/{self.num_goals}: ({self.env.goal_x:.2f}, {self.env.goal_y:.2f})")
        else:
            self.state = next_state
            self.episode_timesteps += 1

    def _print_final_summary(self):
        total_time = time.time() - self.test_start_time
        
        self.get_logger().info("")
        self.get_logger().info("=" * 60)
        self.get_logger().info("           FINAL TEST RESULTS")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"  Model: {self.file_name}")
        self.get_logger().info(f"  Total Goals Tested: {self.num_goals}")
        self.get_logger().info("-" * 60)
        self.get_logger().info(f"  ✓ Goals Reached:  {self.goals_reached}/{self.num_goals} ({100*self.goals_reached/self.num_goals:.1f}%)")
        self.get_logger().info(f"  ✗ Collisions:     {self.collisions}")
        self.get_logger().info(f"  ⏱ Timeouts:       {self.timeouts}")
        self.get_logger().info("-" * 60)
        
        if self.goal_times:
            avg_steps = np.mean(self.goal_times)
            min_steps = np.min(self.goal_times)
            max_steps = np.max(self.goal_times)
            self.get_logger().info(f"  Avg Steps to Goal: {avg_steps:.0f}")
            self.get_logger().info(f"  Min Steps to Goal: {min_steps}")
            self.get_logger().info(f"  Max Steps to Goal: {max_steps}")
        else:
            self.get_logger().info(f"  No goals reached - no step statistics available")
            
        self.get_logger().info("-" * 60)
        self.get_logger().info(f"  Total Test Time:   {total_time:.1f} seconds")
        self.get_logger().info("=" * 60)
        
        # STOP THE ROBOT
        self._stop_robot()
        
        self.get_logger().info("Testing complete. Shutting down...")
        self.get_logger().info("")
    
    def _stop_robot(self):
        """Send zero velocity command to stop the robot."""
        from geometry_msgs.msg import Twist
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.0
        vel_cmd.linear.y = 0.0
        vel_cmd.angular.z = 0.0
        self.env.vel_pub.publish(vel_cmd)
        self.get_logger().info("Robot stopped (cmd_vel = 0)")


def main(args=None):
    rclpy.init(args=args)

    try:
        tester = TD3Tester()
        rclpy.spin(tester)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()