#!/usr/bin/env python3
"""
Live Training Visualizer for TD3

Reads training_log.csv and displays live-updating plots of reward components
and robot state during training.

Usage:
    python3 visualize_live.py [csv_path]
    
If no path provided, defaults to results/training_log.csv
"""
import os
import sys
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configuration
REFRESH_INTERVAL = 2.0  # seconds between updates
WINDOW_SIZE = 1000  # Number of recent timesteps to display


def get_default_csv_path():
    """Get the default path to training_log.csv - checks install dir first, then source."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check source directory first
    source_path = os.path.join(script_dir, "results", "training_log.csv")
    if os.path.exists(source_path):
        return source_path
    
    # Check install directory (where ros2 run creates files)
    # Navigate from source to install path
    install_path = "/home/dark/ros2_ws/src2/Project_PathBlazers/src/install/bots/lib/python3.12/site-packages/bots/td3_rl/results/training_log.csv"
    if os.path.exists(install_path):
        return install_path
    
    # Default to source path (will wait for it to be created)
    return source_path


def load_data(csv_path, max_rows=None):
    """Load the CSV data, handling partial writes gracefully."""
    try:
        if max_rows:
            df = pd.read_csv(csv_path, nrows=max_rows)
        else:
            df = pd.read_csv(csv_path)
        return df
    except pd.errors.EmptyDataError:
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None


def setup_plots():
    """Create the figure and subplots."""
    plt.ion()  # Enable interactive mode
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('TD3 Training Monitor (Live)', fontsize=14, fontweight='bold')
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    return fig, axes


def update_plots(axes, df):
    """Update all subplots with current data."""
    if df is None or len(df) == 0:
        return
    
    # Use recent window
    df_recent = df.tail(WINDOW_SIZE)
    timesteps = df_recent['timestep'].values
    
    # Clear all axes
    for row in axes:
        for ax in row:
            ax.clear()
    
    # Plot 1: Total Reward
    axes[0, 0].plot(timesteps, df_recent['total'].values, 'b-', linewidth=0.8)
    axes[0, 0].set_title('Total Reward')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rolling average overlay
    if len(df_recent) > 50:
        rolling_avg = df_recent['total'].rolling(window=50).mean()
        axes[0, 0].plot(timesteps, rolling_avg.values, 'r-', linewidth=1.5, label='Avg(50)')
        axes[0, 0].legend()
    
    # Plot 2: Reward Components
    reward_cols = ['progress', 'velocity', 'barrier', 'heading']
    colors = ['green', 'blue', 'red', 'orange']
    for col, color in zip(reward_cols, colors):
        if col in df_recent.columns:
            axes[0, 1].plot(timesteps, df_recent[col].values, color=color, linewidth=0.7, label=col, alpha=0.8)
    axes[0, 1].set_title('Reward Components')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].legend(loc='upper right', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Distance to Goal
    if 'distance_to_goal' in df_recent.columns:
        axes[1, 0].plot(timesteps, df_recent['distance_to_goal'].values, 'purple', linewidth=0.8)
        axes[1, 0].set_title('Distance to Goal')
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('Distance (m)')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Minimum Laser Distance (Safety)
    if 'min_laser' in df_recent.columns:
        axes[1, 1].plot(timesteps, df_recent['min_laser'].values, 'red', linewidth=0.8)
        axes[1, 1].axhline(y=0.22, color='darkred', linestyle='--', linewidth=1.5, label='Collision Thresh')
        axes[1, 1].set_title('Minimum Laser Distance')
        axes[1, 1].set_xlabel('Timestep')
        axes[1, 1].set_ylabel('Distance (m)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Control Commands
    cmd_cols = ['cmd_vel_x', 'cmd_vel_y', 'cmd_vel_z']
    cmd_labels = ['Vx', 'Vy', 'Wz']
    cmd_colors = ['blue', 'green', 'red']
    for col, label, color in zip(cmd_cols, cmd_labels, cmd_colors):
        if col in df_recent.columns:
            axes[2, 0].plot(timesteps, df_recent[col].values, color=color, linewidth=0.7, label=label, alpha=0.8)
    axes[2, 0].set_title('Control Commands (cmd_vel)')
    axes[2, 0].set_xlabel('Timestep')
    axes[2, 0].set_ylabel('Velocity')
    axes[2, 0].legend(loc='upper right', fontsize=8)
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 6: Robot Position XY (Trajectory)
    if 'pose_x' in df_recent.columns and 'pose_y' in df_recent.columns:
        # Color by timestep for trajectory visualization
        scatter = axes[2, 1].scatter(
            df_recent['pose_x'].values, 
            df_recent['pose_y'].values, 
            c=timesteps, 
            cmap='viridis', 
            s=2, 
            alpha=0.6
        )
        # Mark current position
        if len(df_recent) > 0:
            axes[2, 1].scatter(
                df_recent['pose_x'].iloc[-1], 
                df_recent['pose_y'].iloc[-1], 
                c='red', s=50, marker='x', linewidths=2, label='Current'
            )
        # Mark goal if available
        if 'goal_x' in df_recent.columns and 'goal_y' in df_recent.columns:
            axes[2, 1].scatter(
                df_recent['goal_x'].iloc[-1], 
                df_recent['goal_y'].iloc[-1], 
                c='lime', s=100, marker='*', linewidths=1, label='Goal'
            )
        axes[2, 1].set_title('Robot Trajectory (XY)')
        axes[2, 1].set_xlabel('X (m)')
        axes[2, 1].set_ylabel('Y (m)')
        axes[2, 1].legend(loc='upper right', fontsize=8)
        axes[2, 1].set_aspect('equal', adjustable='box')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_xlim(-5, 5)
        axes[2, 1].set_ylim(-5, 5)


def main():
    # Get CSV path
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = get_default_csv_path()
    
    print(f"TD3 Training Visualizer")
    print(f"Monitoring: {csv_path}")
    print(f"Refresh interval: {REFRESH_INTERVAL}s")
    print("Press Ctrl+C to exit\n")
    
    # Wait for file to exist
    while not os.path.exists(csv_path):
        print(f"Waiting for {csv_path} to be created...")
        time.sleep(1.0)
    
    fig, axes = setup_plots()
    last_size = 0
    
    try:
        while True:
            # Check if file has been updated
            current_size = os.path.getsize(csv_path) if os.path.exists(csv_path) else 0
            
            if current_size != last_size:
                df = load_data(csv_path)
                if df is not None and len(df) > 0:
                    update_plots(axes, df)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    
                    # Print status
                    latest = df.iloc[-1]
                    print(f"\rTimestep: {int(latest['timestep']):6d} | "
                          f"Episode: {int(latest['episode']):4d} | "
                          f"Reward: {latest['total']:7.2f} | "
                          f"Dist: {latest.get('distance_to_goal', 0):5.2f}m", end='')
                
                last_size = current_size
            
            plt.pause(REFRESH_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\nStopping visualizer...")
    finally:
        plt.ioff()
        plt.close()
        
        # Save final plot
        if 'df' in dir() and df is not None and len(df) > 0:
            output_path = csv_path.replace('.csv', '_plot.png')
            fig_save, axes_save = setup_plots()
            update_plots(axes_save, df)
            fig_save.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Final plot saved to: {output_path}")


if __name__ == "__main__":
    main()
