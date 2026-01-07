#!/usr/bin/env python3
import os
import sys
# Set environment variables to avoid threading conflicts - MUST be before any imports
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF logs
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Prevent OpenBLAS threading issues
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent OpenMP threading issues
os.environ['MKL_NUM_THREADS'] = '1'  # Prevent MKL threading issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' # Allow duplicate OpenMP libs if they occur

# Set multiprocessing start method before any other imports
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# from torch.utils.tensorboard import SummaryWriter

import threading
import time
import csv

# Initialize ROS2 FIRST before any other heavy imports
import rclpy
import faulthandler
faulthandler.enable()
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
except Exception:
    pass

import torch
try:
   torch._dynamo.disable()
except Exception:
   pass
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist
from bots.td3_rl.gazebo_env import GazeboEnv
from bots.td3_rl.replay_buffer import ReplayBuffer



# Global device placeholder - will be set in initialize_environment
device = None


class CSVLogger:
    """Logs training data to a CSV file for visualization."""
    
    def __init__(self, filepath, fieldnames=None):
        self.filepath = filepath
        self.fieldnames = fieldnames or [
            'timestep', 'episode', 'total', 'progress', 'velocity', 'barrier',
            'heading', 'smoothness', 'angular', 'existence', 'min_laser',
            'distance_to_goal', 'cmd_vel_x', 'cmd_vel_y', 'cmd_vel_z',
            'pose_x', 'pose_y', 'pose_yaw', 'goal_x', 'goal_y'
        ]
        self.file = None
        self.writer = None
        self._init_file()
    
    def _init_file(self):
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.file.flush()
    
    def log(self, timestep, episode, info_dict):
        """Log a single timestep's data."""
        row = {'timestep': timestep, 'episode': episode}
        for key in self.fieldnames:
            if key in info_dict:
                row[key] = info_dict[key]
            elif key not in row:
                row[key] = 0.0
        self.writer.writerow(row)
        # Flush periodically for live viewing
        if timestep % 10 == 0:
            self.file.flush()
    
    def close(self):
        if self.file:
            self.file.close()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        # NOTE: Using the original complex layer implementation based on user request.

        s1 = F.relu(self.layer_1(s))
        s1 = F.relu(self.layer_2_s(s1) + self.layer_2_a(a))
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        s2 = F.relu(self.layer_5_s(s2) + self.layer_5_a(a))
        q2 = self.layer_6(s2)
        return q1, q2


# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        td3_rl_path = os.path.dirname(os.path.abspath(__file__))
        
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        
        # DISABLED TensorBoard to prevent glibc crash
        self.writer = None
        self.use_tensorboard = False
        print("Note: TensorBoard is disabled to prevent glibc conflicts.")
        
        self.iter_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=1,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -np.inf
        av_loss = 0
        for it in range(iterations):
            # sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # Obtain the estimated action from the next state by using the actor-target
            next_action = self.actor_target(next_state)

            # Add noise to the action
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss
        self.iter_count += 1
        # Write new values for tensorboard
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
            self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
            self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        """Save actor and critic weights (legacy format for evaluation)."""
        torch.save(self.actor.state_dict(), os.path.join(directory, f"{filename}_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, f"{filename}_critic.pth"))

    def load(self, filename, directory):
        """Load actor and critic weights (legacy format)."""
        self.actor.load_state_dict(torch.load(os.path.join(directory, f"{filename}_actor.pth"), weights_only=False))
        self.critic.load_state_dict(torch.load(os.path.join(directory, f"{filename}_critic.pth"), weights_only=False))
    
    def save_checkpoint(self, filename, directory, metadata=None):
        """Save full training checkpoint including optimizers and metadata."""
        checkpoint = {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'iter_count': self.iter_count,
            'metadata': metadata or {}
        }
        torch.save(checkpoint, os.path.join(directory, f"{filename}_checkpoint.pth"))
    
    def load_checkpoint(self, filename, directory):
        """Load full training checkpoint. Returns metadata dict."""
        checkpoint_path = os.path.join(directory, f"{filename}_checkpoint.pth")
        if not os.path.exists(checkpoint_path):
            return None
        
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.iter_count = checkpoint.get('iter_count', 0)
        return checkpoint.get('metadata', {})


# Provide a minimal DummyNode when ROS is unavailable so the trainer can run in NO_ROS fallback mode
ROS_AVAILABLE = os.environ.get('BOTS_NO_ROS', '0') != '1'

if not ROS_AVAILABLE:
    class DummyLogger:
        def info(self, msg):
            print(msg)
        def warn(self, msg):
            print("WARN:", msg)
        def error(self, msg):
            print("ERROR:", msg)

    class DummyNode:
        def __init__(self, name=None):
            self._parameters = {}
            self._logger = DummyLogger()
        def declare_parameter(self, name, value=None):
            self._parameters[name] = value
        def get_parameter(self, name):
            class P:
                pass
            p = P()
            p.value = self._parameters.get(name)
            return p
        def get_logger(self):
            return self._logger
        def destroy_node(self):
            pass

    BaseNode = DummyNode
else:
    BaseNode = Node

class TD3Trainer(BaseNode):
    def __init__(self):
        # If real ROS is available, initialize Node via super(); otherwise init DummyNode
        if ROS_AVAILABLE:
            super().__init__('td3_trainer')
        else:
            BaseNode.__init__(self, 'td3_trainer')
        self.declare_params()
        self._shutdown_flag = False  # Flag to signal executor thread to stop
        self._executor_thread = None  # Store the executor thread reference
        # The executor is now saved to self.executor for later use
        self.executor = self.initialize_environment()
        self.train_loop()

    def declare_params(self):
        self.declare_parameter('seed', 0)
        self.declare_parameter('eval_freq', 500)
        self.declare_parameter('max_ep', 150)
        self.declare_parameter('eval_ep', 10)
        self.declare_parameter('max_timesteps', 500000)
        self.declare_parameter('expl_noise', 0.8)
        self.declare_parameter('expl_decay_steps', 100000)
        self.declare_parameter('expl_min', 0.05)
        self.declare_parameter('batch_size', 128)
        self.declare_parameter('discount', 0.99)
        self.declare_parameter('tau', 0.005)
        self.declare_parameter('policy_noise', 0.2)
        self.declare_parameter('noise_clip', 0.5)
        self.declare_parameter('policy_freq', 2)
        self.declare_parameter('buffer_size', 1000000)
        self.declare_parameter('file_name', 'TD3_Mecanum')
        self.declare_parameter('save_model', True)
        self.declare_parameter('load_model', False)
        self.declare_parameter('random_near_obstacle', True)
        self.declare_parameter('environment_dim', 20)
        
        # Ground-truth pose parameters
        self.declare_parameter('use_ground_truth', True)
        self.declare_parameter('pose_topic', '/model/my_robot/odometry_with_covariance')
        self.declare_parameter('odom_topic', '/odometry/filtered')
        self.declare_parameter('ground_truth_noise_std', 0.0)
    
    def print_parameters(self):
        self.get_logger().info("Loaded Parameters:")
        for param in self._parameters.keys():
            self.get_logger().info(f"{param}: {self.get_parameter(param).value}")

    def initialize_environment(self):
        # --- THREADING SAFEGUARDS ---
        global device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.get_logger().info(f"Using device: {self.device}")
        
        # Enforce single-threaded execution for PyTorch to avoid ROS 2 conflicts
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        # ----------------------------

        self.seed = self.get_parameter('seed').value
        self.eval_freq = self.get_parameter('eval_freq').value
        self.max_ep = self.get_parameter('max_ep').value
        self.eval_ep = self.get_parameter('eval_ep').value
        self.max_timesteps = self.get_parameter('max_timesteps').value
        self.expl_noise = self.get_parameter('expl_noise').value
        self.expl_decay_steps = self.get_parameter('expl_decay_steps').value
        self.expl_min = self.get_parameter('expl_min').value
        self.batch_size = self.get_parameter('batch_size').value
        self.discount = self.get_parameter('discount').value
        self.tau = self.get_parameter('tau').value
        self.policy_noise = self.get_parameter('policy_noise').value
        self.noise_clip = self.get_parameter('noise_clip').value
        self.policy_freq = self.get_parameter('policy_freq').value
        self.buffer_size = self.get_parameter('buffer_size').value
        self.file_name = self.get_parameter('file_name').value
        self.save_model = self.get_parameter('save_model').value
        self.load_model = self.get_parameter('load_model').value
        self.random_near_obstacle = self.get_parameter('random_near_obstacle').value
        self.environment_dim = self.get_parameter('environment_dim').value
        
        self.use_ground_truth = self.get_parameter('use_ground_truth').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.ground_truth_noise_std = self.get_parameter('ground_truth_noise_std').value
        
        self.print_parameters()

        # NOTE: Using a placeholder path here. Replace with your actual path if needed.
        td3_rl_path = os.path.dirname(os.path.abspath(__file__)) 
        
        self.results_path = os.path.join(td3_rl_path, "results")
        self.models_path = os.path.join(td3_rl_path, "pytorch_models")
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)

        self.env = GazeboEnv(
            self.environment_dim,
            use_ground_truth=self.use_ground_truth,
            pose_topic=self.pose_topic,
            odom_topic=self.odom_topic,
            ground_truth_noise_std=self.ground_truth_noise_std
        )
        
        # --- MODIFICATION: Create one MultiThreadedExecutor for both nodes ---
        executor = MultiThreadedExecutor()
        # Add environment node (for subscriptions) and trainer node (for logger/services)
        executor.add_node(self.env)
        executor.add_node(self) # Add the TD3Trainer node itself
        
        # Run the executor in a separate thread so the main thread can run the RL logic
        def spin_executor():
            try:
                while rclpy.ok() and not self._shutdown_flag:
                    executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                if not self._shutdown_flag:
                    self.get_logger().error(f"Executor error: {e}")
        
        self._executor_thread = threading.Thread(target=spin_executor, daemon=True)
        self._executor_thread.start()
        
        time.sleep(5)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # --- EXPLICIT CUDA INITIALIZATION ---
        if self.device.type == 'cuda':
             try:
                 torch.cuda.init()
                 # Create a dummy tensor to force initialization
                 _ = torch.tensor([0.0], device=self.device)
             except Exception as e:
                 self.get_logger().warn(f"CUDA init warning: {e}")
        # ------------------------------------

        
        # --- CRITICAL CHANGES FOR MECANUM ACTION/STATE DIMENSIONS ---
        self.action_dim = 3  
        # State: 20 LIDAR rays + 9 robot states (distance, sin(theta), cos(theta), 
        #        vx, vy, omega, front_danger, left_danger, right_danger)
        self.state_dim = self.environment_dim + 9
        self.max_action = 1 
        # -----------------------------------------------------------

        self.network = TD3(self.state_dim, self.action_dim, self.max_action)
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.seed)
        
        # Note: Model loading is now handled in train_loop for full checkpoint resume support
        
        return executor # Return the executor to be shut down later

    def train_loop(self):
        evaluations = []
        timestep, timesteps_since_eval, episode_num = 0, 0, 0
        done = True
        epoch = 1
        last_time = time.time()
        
        count_rand_actions = 0
        episode_timesteps = 0
        episode_reward = 0  # Initialize to prevent error on first done check when resuming
        random_action = np.zeros(self.action_dim) 
        last_time = time.time()
        
        self.best_avg_reward = -np.inf # Initialize best reward tracking
        
        # --- METRICS TRACKING ---
        self.total_goals_reached = 0
        self.total_collisions = 0
        self.recent_rewards = []  # Track last 10 episode rewards
        
        # --- Goal Stability Status ---
        goal_reached_last_step = False
        
        # --- INITIALIZE CSV LOGGER ---
        csv_log_path = os.path.join(self.results_path, "training_log.csv")
        self.csv_logger = CSVLogger(csv_log_path)
        self.get_logger().info(f"CSV logging to: {csv_log_path}")
        
        # --- CHECKPOINT RESUME LOGIC ---
        if self.load_model:
            # Try to load full checkpoint first
            replay_buffer_path = os.path.join(self.models_path, f"{self.file_name}_replay.pkl")
            metadata = self.network.load_checkpoint(self.file_name, self.models_path)
            
            if metadata is not None:
                timestep = metadata.get('timestep', 0)
                episode_num = metadata.get('episode_num', 0)
                epoch = metadata.get('epoch', 1)
                self.expl_noise = metadata.get('expl_noise', self.expl_noise)
                self.best_avg_reward = metadata.get('best_avg_reward', -np.inf)
                self.total_goals_reached = metadata.get('total_goals_reached', 0)
                self.total_collisions = metadata.get('total_collisions', 0)
                
                # Load replay buffer if exists
                if os.path.exists(replay_buffer_path):
                    self.replay_buffer.load(replay_buffer_path)
                    self.get_logger().info(f"Loaded replay buffer with {self.replay_buffer.size()} experiences")
                
                self.get_logger().info(f"Resumed from checkpoint: timestep={timestep}, episode={episode_num}, epoch={epoch}")
            else:
                # Fallback to legacy load (just weights)
                try:
                    self.network.load(self.file_name, self.models_path)
                    self.get_logger().info("Loaded model weights (legacy format). Starting from timestep 0.")
                except Exception:
                    self.get_logger().warn("Could not load model, initializing training with random parameters")

        try:
            while timestep < self.max_timesteps:
                if timestep % 100 == 0:
                    current_time = time.time()
                    self.get_logger().info(f"{timestep} timesteps. Last 100 timesteps finished in {(current_time - last_time):.2f} seconds")
                    last_time = current_time
                
                if done:
                    if timestep != 0:
                        self.network.train(
                            self.replay_buffer,
                            episode_timesteps,
                            self.batch_size,
                            self.discount,
                            self.tau,
                            self.policy_noise,
                            self.noise_clip,
                            self.policy_freq,
                        )
                    if timesteps_since_eval >= self.eval_freq:
                        self.get_logger().info(f"Validating at timestep {timestep}")
                        timesteps_since_eval %= self.eval_freq
                        
                        # LOGGING EVALUATION TO TENSORBOARD 
                        avg_reward_eval, avg_col_eval = self.evaluate(network=self.network, epoch=epoch, eval_episodes=self.eval_ep)
                        if self.network.use_tensorboard and self.network.writer is not None:
                            self.network.writer.add_scalar("Evaluation/Avg_Reward", avg_reward_eval, timestep)
                            self.network.writer.add_scalar("Evaluation/Collision_Rate", avg_col_eval, timestep)
                        
                        evaluations.append(avg_reward_eval)
                        
                        if self.save_model:
                            self.network.save(self.file_name, directory=self.models_path)
                            np.save(os.path.join(self.results_path, f"{self.file_name}.npy"), evaluations)
                            
                            # --- SAVE BEST MODEL ---
                            if avg_reward_eval > self.best_avg_reward:
                                self.best_avg_reward = avg_reward_eval
                                self.get_logger().info(f"New best model found! Reward: {avg_reward_eval:.2f}. Saving...")
                                self.network.save(f"{self.file_name}_best", directory=self.models_path)
                            # -----------------------
                            
                        epoch += 1
                    
                    # --- GOAL STABILITY FIX: Update goal status BEFORE reset ---
                    is_timeout = (episode_timesteps + 1 == self.max_ep)
                    
                    # SAVE the flag BEFORE resetting (for logging)
                    episode_reached_goal = goal_reached_last_step
                    
                    if hasattr(self.env, 'update_goal_status'):
                        self.env.update_goal_status(goal_reached_last_step, is_timeout)
                    goal_reached_last_step = False 
                    
                    state = self.env.reset()
                    # --------------------------------------------------------
                    
                    # --- LOG EPISODE METRICS ---
                    if timestep != 0:
                        outcome = 'GOAL' if episode_reached_goal else ('COLLISION' if episode_reward < -100 else 'TIMEOUT')
                        if episode_reached_goal:
                            self.total_goals_reached += 1
                        elif episode_reward < -100:
                            self.total_collisions += 1
                        
                        self.recent_rewards.append(episode_reward)
                        if len(self.recent_rewards) > 10:
                            self.recent_rewards.pop(0)
                        avg_recent = np.mean(self.recent_rewards)
                        
                        self.get_logger().info(
                            f"Ep {episode_num} | Reward: {episode_reward:.1f} | Outcome: {outcome} | "
                            f"Goals: {self.total_goals_reached} | Collisions: {self.total_collisions} | "
                            f"Avg(10): {avg_recent:.1f} | Noise: {self.expl_noise:.3f}"
                        )
                    # ---------------------------
                    
                    done = False
                    episode_reward = 0
                    episode_timesteps = 0
                    episode_num += 1
                
                # add some exploration noise
                if self.expl_noise > self.expl_min:
                    self.expl_noise = self.expl_noise - ((1 - self.expl_min) / self.expl_decay_steps)
                
                action = self.network.get_action(np.array(state))
                action = (action + np.random.normal(0, self.expl_noise, size=self.action_dim)).clip(
                    -self.max_action, self.max_action)
                
                # --- CORRECTED OBSTACLE EXPLORATION LOGIC: REMOVED FORCED REVERSE ---
                if self.random_near_obstacle:
                    if (
                        np.random.uniform(0, 1) > 0.85
                        and min(state[4:-8]) < 0.6 # Check for proximity to obstacles
                        and count_rand_actions < 1
                    ):
                        count_rand_actions = np.random.randint(8, 15)
                        random_action = np.random.uniform(-self.max_action, self.max_action, self.action_dim) 

                    if count_rand_actions > 0:
                        count_rand_actions -= 1
                        action = random_action
                        # No more forced reverse action[0] = -1
                # ---------------------------------------------
                
                # --- ACTION SCALING FIX: FULL RANGE [-1, 1] FOR LIN X ---
                a_in = [
                    action[0],            # Lin X: Full range [-1, 1] (Forward and Backward)
                    action[1],            # Lin Y: Keep [-1, 1] (Strafe left/right)
                    action[2]             # Ang Z: Keep [-1, 1] (Rotate CW/CCW)
                ]
                # --------------------------------------------------------
                
                next_state, reward, done, target, step_info = self.env.step(a_in)
                
                # --- LOG TO CSV ---
                self.csv_logger.log(timestep, episode_num, step_info)
                
                # --- GOAL STABILITY FIX: Set flag if goal was reached in this step ---
                if target:
                    goal_reached_last_step = True
                    self.get_logger().info('GOAL REACHED! Updating goal on next reset.')
                # ---------------------------------------------------------------------
                
                done_bool = 0 if episode_timesteps + 1 == self.max_ep else int(done)
                done = 1 if episode_timesteps + 1 == self.max_ep else int(done)
                episode_reward += reward
                
                # Save the tuple in replay buffer
                self.replay_buffer.add(state, action, reward, done_bool, next_state)

                state = next_state
                episode_timesteps += 1
                timestep += 1
                timesteps_since_eval += 1

        # --- GRACEFUL SHUTDOWN AND SAVE ---
        except KeyboardInterrupt:
            self.get_logger().info("Caught CTRL+C. Shutting down gracefully and saving checkpoint...")
            self._stop_robot()  # Stop the robot
            
            # Save full checkpoint
            if self.save_model:
                metadata = {
                    'timestep': timestep,
                    'episode_num': episode_num,
                    'epoch': epoch,
                    'expl_noise': self.expl_noise,
                    'best_avg_reward': self.best_avg_reward,
                    'total_goals_reached': self.total_goals_reached,
                    'total_collisions': self.total_collisions
                }
                self.network.save_checkpoint(self.file_name, self.models_path, metadata)
                self.replay_buffer.save(os.path.join(self.models_path, f"{self.file_name}_replay.pkl"))
                np.save(os.path.join(self.results_path, f"{self.file_name}.npy"), evaluations)
                self.get_logger().info(f"Full checkpoint saved at timestep {timestep}.")
            
            # Close CSV logger
            if hasattr(self, 'csv_logger'):
                self.csv_logger.close()
        # --- END GRACEFUL SHUTDOWN ---

        # After the training is done (either by max_timesteps or interruption), perform final save
        if timestep < self.max_timesteps and self.save_model:
            # If interrupted, we already saved, so we just pass to clean up the executor.
            pass 
        elif timestep >= self.max_timesteps:
            # Only save and evaluate if the max timesteps was hit
            self._stop_robot()  # Stop the robot
            eval_result = self.evaluate(network=self.network, epoch=epoch, eval_episodes=self.eval_ep)
            evaluations.append(eval_result[0] if isinstance(eval_result, tuple) else eval_result)
            if self.save_model:
                metadata = {
                    'timestep': timestep,
                    'episode_num': episode_num,
                    'epoch': epoch,
                    'expl_noise': self.expl_noise,
                    'best_avg_reward': self.best_avg_reward,
                    'total_goals_reached': self.total_goals_reached,
                    'total_collisions': self.total_collisions
                }
                self.network.save_checkpoint(self.file_name, self.models_path, metadata)
                self.network.save(self.file_name, directory=self.models_path)  # Also save legacy format
                self.replay_buffer.save(os.path.join(self.models_path, f"{self.file_name}_replay.pkl"))
                np.save(os.path.join(self.results_path, f"{self.file_name}.npy"), evaluations)
            
            # Close CSV logger
            if hasattr(self, 'csv_logger'):
                self.csv_logger.close()
    
    def _stop_robot(self):
        """Send zero velocity command to stop the robot."""
        try:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.linear.y = 0.0
            vel_cmd.angular.z = 0.0
            self.env.vel_pub.publish(vel_cmd)
            self.get_logger().info("Robot stopped (cmd_vel = 0)")
        except Exception:
            # ROS context might be shutting down, ignore publish errors
            pass


    def evaluate(self, network, epoch, eval_episodes=10):
        avg_reward = 0.0
        col = 0
        for _ in range(eval_episodes):
            count = 0
            state = self.env.reset()
            done = False
            while not done and count < self.max_ep:
                action = network.get_action(np.array(state))
                
                # --- ACTION SCALING FIX IN EVALUATION: FULL RANGE [-1, 1] FOR LIN X ---
                a_in = [
                    action[0],           # Lin X: Full range [-1, 1]
                    action[1],           # Lin Y: Keep [-1, 1]
                    action[2]            # Ang Z: Keep [-1, 1]
                ]
                # -----------------------------------------------------

                state, reward, done, _, _ = self.env.step(a_in)
                avg_reward += reward
                count += 1
                if reward < -90:
                    col += 1
        avg_reward /= eval_episodes
        avg_col = col / eval_episodes
        self.get_logger().info("..............................................")
        self.get_logger().info(
            f"Average Reward over {eval_episodes} Evaluation Episodes, Epoch {epoch}: {avg_reward:.2f}, Collision Rate: {avg_col:.2f}"
        )
        self.get_logger().info("..............................................")
        return avg_reward, avg_col


# Removed _check_rcl_node_viable as it uses subprocess/fork which can trigger glibc errors
# when ML libraries (torch/tensorflow) are already initialized in the parent process.


def main(args=None):
    node = None

    # Initialize rclpy directly. Subprocess check removed to avoid glibc conflicts.
    try:
        rclpy.init(args=args)
    except Exception as e:
        print(f"Warning: rclpy.init() failed: {e}. Proceeding in NO_ROS mode.")
        os.environ['BOTS_NO_ROS'] = '1'

    try:
        node = TD3Trainer()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # --- MODIFICATION: Properly shutdown the executor and cleanup ---
        if node is not None:
            # Signal the executor thread to stop
            node._shutdown_flag = True

            # Attempt to ask executor to shutdown (may help wake blocking calls)
            try:
                if hasattr(node, 'executor') and node.executor is not None:
                    node.get_logger().info("Shutting down executor (pre-join)")
                    try:
                        node.executor.shutdown()
                    except Exception as e:
                        node.get_logger().warn(f"Executor shutdown (pre-join) raised: {e}")
            except Exception:
                pass

            # Wait for executor thread to finish (with longer timeout)
            if node._executor_thread is not None and node._executor_thread.is_alive():
                node._executor_thread.join(timeout=5.0)

            # Remove nodes from executor before final cleanup
            try:
                if hasattr(node, 'executor') and node.executor is not None:
                    try:
                        node.executor.remove_node(node.env)
                    except Exception:
                        pass
                    try:
                        node.executor.remove_node(node)
                    except Exception:
                        pass
            except Exception:
                pass

            # Final executor shutdown (best-effort)
            try:
                if hasattr(node, 'executor') and node.executor is not None:
                    node.get_logger().info("Final executor.shutdown()")
                    try:
                        node.executor.shutdown()
                    except Exception as e:
                        node.get_logger().warn(f"Executor shutdown (final) raised: {e}")
            except Exception:
                pass

            # Destroy nodes
            try:
                node.env.destroy_node()
            except Exception:
                pass
            try:
                node.destroy_node()
            except Exception:
                pass
        
        # Shutdown rclpy
        try:
            rclpy.shutdown()
        except Exception:
            pass
        # -----------------------------------------------------------------------------


if __name__ == '__main__':
    main()