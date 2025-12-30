#!/usr/bin/env python3
import os
import threading
import time
import sys # Import sys for clean exit

import numpy as np
import rclpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from geometry_msgs.msg import Twist
from bots.td3_rl.gazebo_env import GazeboEnv
from bots.td3_rl.replay_buffer import ReplayBuffer


from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
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
        self.writer = SummaryWriter(log_dir=os.path.join(td3_rl_path, "runs"))
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
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), os.path.join(directory, f"{filename}_actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, f"{filename}_critic.pth"))

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(os.path.join(directory, f"{filename}_actor.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(directory, f"{filename}_critic.pth")))


class TD3Trainer(Node):
    def __init__(self):
        super().__init__('td3_trainer')
        self.declare_params()
        # The executor is now saved to self.executor for later use
        self.executor = self.initialize_environment() 
        self.train_loop()

    def declare_params(self):
        self.declare_parameter('seed', 0)
        self.declare_parameter('eval_freq', 500)
        self.declare_parameter('max_ep', 150)
        self.declare_parameter('eval_ep', 10)
        self.declare_parameter('max_timesteps', 500000)
        self.declare_parameter('expl_noise', 1.0)
        self.declare_parameter('expl_decay_steps', 500000)
        self.declare_parameter('expl_min', 0.1)
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
    
    def print_parameters(self):
        self.get_logger().info("Loaded Parameters:")
        for param in self._parameters.keys():
            self.get_logger().info(f"{param}: {self.get_parameter(param).value}")

    def initialize_environment(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        
        self.print_parameters()

        # NOTE: Using a placeholder path here. Replace with your actual path if needed.
        td3_rl_path = os.path.dirname(os.path.abspath(__file__)) 
        
        self.results_path = os.path.join(td3_rl_path, "results")
        self.models_path = os.path.join(td3_rl_path, "pytorch_models")
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)

        self.env = GazeboEnv(self.environment_dim)
        
        # --- MODIFICATION: Create one MultiThreadedExecutor for both nodes ---
        executor = MultiThreadedExecutor()
        # Add environment node (for subscriptions) and trainer node (for logger/services)
        executor.add_node(self.env)
        executor.add_node(self) # Add the TD3Trainer node itself
        
        # Run the executor in a separate thread so the main thread can run the RL logic
        env_thread = threading.Thread(target=executor.spin, daemon=True)
        env_thread.start()
        
        time.sleep(5)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # --- CRITICAL CHANGES FOR MECANUM ACTION/STATE DIMENSIONS ---
        self.action_dim = 3  
        self.state_dim = self.environment_dim + 5 
        self.max_action = 1 
        # -----------------------------------------------------------

        self.network = TD3(self.state_dim, self.action_dim, self.max_action)
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.seed)
        
        if self.load_model:
            try:
                self.network.load(self.file_name, self.models_path)
            except:
                self.get_logger().warn("Could not load model, initializing training with random parameters")
        
        return executor # Return the executor to be shut down later

    def train_loop(self):
        evaluations = []
        timestep, timesteps_since_eval, episode_num = 0, 0, 0
        done = True
        epoch = 1
        last_time = time.time()
        
        count_rand_actions = 0
        episode_timesteps = 0
        random_action = np.zeros(self.action_dim) 
        last_time = time.time()
        
        self.best_avg_reward = -np.inf # Initialize best reward tracking
        
        # --- METRICS TRACKING ---
        self.total_goals_reached = 0
        self.total_collisions = 0
        self.recent_rewards = []  # Track last 10 episode rewards
        
        # --- Goal Stability Status ---
        goal_reached_last_step = False

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
                
                next_state, reward, done, target = self.env.step(a_in)
                
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
            self.get_logger().info("Caught CTRL+C. Shutting down gracefully and saving model...")
            self._stop_robot()  # Stop the robot
            # Save the evaluations array up to the point of interruption
            np.save(os.path.join(self.results_path, f"{self.file_name}_interrupted.npy"), evaluations)
            if self.save_model:
                self.network.save(f"{self.file_name}_interrupted", directory=self.models_path)
                self.get_logger().info(f"Model and evaluation results saved at timestep {timestep}.")
            # Now, exit the loop and continue to final save block
        # --- END GRACEFUL SHUTDOWN ---

        # After the training is done (either by max_timesteps or interruption), perform final save
        if timestep < self.max_timesteps and self.save_model:
            # If interrupted, we already saved, so we just pass to clean up the executor.
            pass 
        elif timestep >= self.max_timesteps:
            # Only save and evaluate if the max timesteps was hit
            self._stop_robot()  # Stop the robot
            evaluations.append(self.evaluate(network=self.network, epoch=epoch, eval_episodes=self.eval_ep))
            if self.save_model:
                self.network.save(self.file_name, directory=self.models_path)
                np.save(os.path.join(self.results_path, f"{self.file_name}.npy"), evaluations)
    
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

                state, reward, done, _ = self.env.step(a_in)
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


def main(args=None):
    rclpy.init(args=args)
    node = TD3Trainer()
    # --- MODIFICATION: Shutdown the executor returned by initialize_environment ---
    # The training thread is already running the executor's spin in the background.
    # We now call shutdown on the executor when the main node finishes its train_loop.
    node.executor.shutdown()
    # -----------------------------------------------------------------------------
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()