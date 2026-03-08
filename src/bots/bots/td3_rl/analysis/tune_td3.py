#!/usr/bin/env python3
import os
import yaml
import time
import numpy as np
import rclpy

# Re-use our core Simulation Environment and TD3 algorithms
from bots.td3_rl.train_td3 import TD3Trainer
from rclpy.executors import MultiThreadedExecutor

class HyperparameterTuner:
    def __init__(self):
        self.td3_rl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self.analysis_path = os.path.join(self.td3_rl_path, "analysis")
        os.makedirs(self.analysis_path, exist_ok=True)
        
        # Grid Search Definitions
        self.param_grid = {
            'batch_size': [128, 256, 512],
            'tau': [0.005, 0.001],
            'policy_noise': [0.1, 0.2, 0.3],
            'discount': [0.99, 0.995]
        }
        
        self.best_reward = -np.inf
        self.best_params = None

    def run_tuning(self):
        print("\n" + "="*50)
        print("STARTING AUTOMATED HYPERPARAMETER TUNING")
        print("="*50 + "\n")

        combinations = [
            {'batch_size': b, 'tau': t, 'policy_noise': p, 'discount': d}
            for b in self.param_grid['batch_size']
            for t in self.param_grid['tau']
            for p in self.param_grid['policy_noise']
            for d in self.param_grid['discount']
        ]
        
        # Shuffle for random search approximation
        np.random.shuffle(combinations)
        
        # Run top 5 combos (Time constraints: 5 * 5min = 25 minutes of tuning)
        combinations = combinations[:5]

        for i, config in enumerate(combinations):
            print(f"\n[EVALUATING CONFIG {i+1}/{len(combinations)}]")
            print(f"Params: {config}")

            # Create Trainer instance natively
            trainer = TD3Trainer()
            
            # --- OVERRIDE PARAMETERS FOR SHORT BURST ---
            # Roughly 5 minutes of training (750 timesteps at simulation speed)
            trainer.max_timesteps = 750 
            trainer.eval_freq = 750
            trainer.max_ep = 150
            trainer.save_model = False # Don't clutter with bad models
            trainer.file_name = f"TD3_Tuning_Config_{i}"
            
            # Apply our Grid Search values
            trainer.batch_size = config['batch_size']
            trainer.tau = config['tau']
            trainer.policy_noise = config['policy_noise']
            trainer.discount = config['discount']
            
            # Recreate Replay Buffer/Network with new sizing if needed
            from bots.td3_rl.replay_buffer import ReplayBuffer
            from bots.td3_rl.train_td3 import TD3
            trainer.network = TD3(trainer.state_dim, trainer.action_dim, trainer.max_action)
            trainer.replay_buffer = ReplayBuffer(trainer.buffer_size, trainer.seed)

            # --- RUN SHORT EPOCH ---
            start_time = time.time()
            trainer.train_loop()
            end_time = time.time()
            
            # --- FLAW ANALYSIS ---
            total_episodes = trainer.total_goals_reached + trainer.total_collisions
            if total_episodes == 0: total_episodes = 1 
            collision_rate = trainer.total_collisions / total_episodes
            
            avg_reward = np.mean(trainer.recent_rewards) if trainer.recent_rewards else -1000
            
            print(f"--- RESULTS CONFIG {i+1} ---")
            print(f"Time Taken: {(end_time - start_time):.1f}s")
            print(f"Goals: {trainer.total_goals_reached} | Collisions: {trainer.total_collisions}")
            print(f"Collision Rate: {collision_rate*100:.1f}% | Avg Reward: {avg_reward:.1f}")

            # Check for catastrophic flaws (> 20% collisions is discarded)
            if collision_rate > 0.20:
                print(f"FLAW DETECTED: Collision rate {collision_rate*100:.1f}% exceeds 20% threshold. Discarding configuration.")
            else:
                if avg_reward > self.best_reward:
                    print("NEW BEST CONFIGURATION FOUND!")
                    self.best_reward = avg_reward
                    self.best_params = config
                    self.save_best(config, avg_reward, collision_rate)
            
            # Shutdown ROS node before looping
            try:
                trainer.destroy_node()
            except: pass

        print("\n" + "="*50)
        print("TUNING COMPLETE.")
        print(f"Best Configuration Saved to analysis folder.")
        print("="*50 + "\n")

    def save_best(self, config, reward, collision_rate):
        filepath = os.path.join(self.analysis_path, "best_hyperparams.yaml")
        data = {
            'best_training_metrics': {
                'avg_reward': float(reward),
                'collision_rate': float(collision_rate)
            },
            'hyperparameters': config
        }
        with open(filepath, 'w') as f:
            yaml.dump(data, f)


def main(args=None):
    rclpy.init(args=args)
    tuner = HyperparameterTuner()
    try:
        tuner.run_tuning()
    except KeyboardInterrupt:
        print("Tuning interrupted by user.")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
