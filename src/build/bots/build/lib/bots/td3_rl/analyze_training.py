#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def analyze_results(file_name="TD3_Mecanum"):
    # Use relative path: script_dir/results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(script_dir, "results")
    file_path = os.path.join(results_path, f"{file_name}.npy")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print("Available files in results folder:")
        print(os.listdir(results_path))
        return

    print(f"Loading results from: {file_path}")
    evaluations = np.load(file_path)
    
    if len(evaluations) == 0:
        print("No evaluation data found.")
        return

    print("\n--- Training Analysis ---")
    print(f"Total Evaluations: {len(evaluations)}")
    print(f"Max Reward: {np.max(evaluations):.2f}")
    print(f"Min Reward: {np.min(evaluations):.2f}")
    print(f"Average Reward (All): {np.mean(evaluations):.2f}")
    
    if len(evaluations) >= 10:
        print(f"Average Reward (Last 10): {np.mean(evaluations[-10:]):.2f}")
    else:
        print(f"Average Reward (Last {len(evaluations)}): {np.mean(evaluations):.2f}")

    # Identify "Good" and "Bad" epochs
    good_threshold = 0  # Positive reward is generally good in this task
    bad_threshold = -150 # Close to collision penalty
    
    good_epochs = np.where(evaluations > good_threshold)[0]
    bad_epochs = np.where(evaluations < bad_threshold)[0]
    
    print(f"\nGood Epochs (> {good_threshold}): {len(good_epochs)}")
    print(f"Bad Epochs (< {bad_threshold}): {len(bad_epochs)}")

    # Plotting
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(evaluations, label='Avg Reward')
        plt.axhline(y=0, color='g', linestyle='--', label='Zero Reward')
        plt.axhline(y=-200, color='r', linestyle='--', label='Collision Baseline')
        plt.title(f"Training Progress - {file_name}")
        plt.xlabel("Evaluation Epochs (x5000 steps)")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(results_path, f"{file_name}_plot.png")
        plt.savefig(plot_path)
        print(f"\nPlot saved to: {plot_path}")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_results(sys.argv[1])
    else:
        analyze_results()
