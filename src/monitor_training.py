import os
import time
import subprocess
import signal
import sys
from datetime import datetime

# --- CONFIGURATION ---
# Use the src directory as CWD
WORKSPACE_DIR = "/home/dark/ros2_ws/src2/Project_PathBlazers/src"
SOURCE_CMD = "source /home/dark/ros2_ws/src2/Project_PathBlazers/install/setup.bash"

# Flags: load_model=True to resume, save_model=True to save progress
# Note: We must restart the LAUNCH file (sim) AND the NODE (trainer).
# The most reliable way is to launch the standard launch file, and ensuring the trainer node picks up the args.
# Since 'train_td3.launch.py' launches 'spawn_robot.launch.py' but might not include the python trainer node itself 
# (based on previous observations where we had to run 'ros2 run bots train_td3' separately or maybe it was added?),
# let's look at the previous 'train_td3.launch.py': It definitely did NOT include the 'td3_trainer' node.
# So we need a compound command.

TRAINER_CMD = "ros2 run bots train_td3 --ros-args -p load_model:=True -p save_model:=True"
# Explicitly enable Gazebo GUI so watchdog launches with GUI visible
LAUNCH_CMD = "ros2 launch bots train_td3.launch.py gui:=true"

# Compound command:
# 1. Source setup.bash
# 2. Launch Gazebo in background checking for success
# 3. Wait for it to settle
# 4. Launch Trainer
# We wrap this in a shell script string.
FULL_CMD = f"/bin/bash -c '{SOURCE_CMD} && {LAUNCH_CMD} & sleep 15 && {SOURCE_CMD} && {TRAINER_CMD}'"

LOG_FILE = "training_long.log" # Relative to CWD
REPORT_FILE = "monitor_report.txt"

# Watchdog Settings
STUCK_PHRASE = "Pose update TIMEOUT! Physics likely stuck"
STUCK_THRESHOLD = 5  # Number of consecutive stuck messages to trigger restart (Aggressive)
RESET_COOLDOWN = 30   # Seconds to wait for startup before monitoring starts

def log_msg(msg):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    formatted_msg = f"{timestamp} {msg}"
    print(formatted_msg)
    try:
        with open(REPORT_FILE, "a") as f:
            f.write(formatted_msg + "\n")
    except Exception as e:
        print(f"Error writing to report: {e}")

def kill_all_processes():
    log_msg("⚠️ KILLING ALL ROS/GAZEBO PROCESSES...")
    # Force kill everything related to our session
    commands = [
        "pkill -9 -f 'train_td3'",
        "pkill -9 -f 'gz'",
        "pkill -9 -f 'ruby'",
        "pkill -9 -f 'ros2'",
        "pkill -9 -f 'python3.*train_td3'",
        "pkill -9 -f 'spawn_robot'",
        "pkill -9 -f 'robot_state_publisher'"
    ]
    for cmd in commands:
        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5) # Cleanup time

def run_watchdog():
    log_msg("--- WATCHDOG STARTED ---")
    
    while True:
        # 1. Clean Slate
        kill_all_processes()
        
        # 2. Start the Training Stack
        log_msg("🚀 Launching Training Stack...")
        
        # Start the process, redirecting ALL output to the log file for us to read.
        # using setsid to allow killing the whole group if needed.
        with open(LOG_FILE, "w") as log_f:
            process = subprocess.Popen(
                FULL_CMD, 
                shell=True, 
                cwd=WORKSPACE_DIR,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
        
        log_msg(f"Process PID: {process.pid}. Waiting {RESET_COOLDOWN}s for startup...")
        time.sleep(RESET_COOLDOWN)
        
        # 3. Monitor Loop
        log_msg("👀 Monitoring logs for errors...")
        stuck_count = 0
        success_count = 0
        
        # Open the file for reading (tail mode)
        try:
            f = open(os.path.join(WORKSPACE_DIR, LOG_FILE), "r")
            f.seek(0, 2) # Seek to end
        except FileNotFoundError:
            log_msg("Error: Log file not found! waiting and retrying...")
            time.sleep(5)
            continue

        monitoring_active = True
        while monitoring_active:
            # Check if process is still alive first
            if process.poll() is not None:
                log_msg("❌ Main process died unexpectedly! Restarting in 5s...")
                time.sleep(5)
                monitoring_active = False
                break

            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
            
            # Sanity Check for Physics Freeze
            if STUCK_PHRASE in line:
                stuck_count += 1
                success_count = 0 # Reset success streak
                if stuck_count % 5 == 0:
                    print(f"Warning: Physics stuck detected ({stuck_count}/{STUCK_THRESHOLD})")
            elif "CB" in line or "Reward" in line or "[STEP]" in line:
                # If we see normal step logs (that aren't timeouts), we are alive.
                # However, sometimes logs interleave.
                # Let's say if we get a "Reward" or explicit position update, we reset the stuck counter.
                if "Pose update TIMEOUT" not in line:
                     success_count += 1
                     if success_count > 5:
                         stuck_count = 0 # We are healthy
            
            # Trigger Restart
            if stuck_count >= STUCK_THRESHOLD:
                log_msg(f"🚨 CRITICAL: Physics Frozen detected ({stuck_count} warnings). TRIGGERING RESTART.")
                monitoring_active = False 
                break
            
            # Log periodic status to console to show liveness
            if "Ep" in line and "Reward" in line:
                print(line.strip())
                stuck_count = 0

        # Close file before looping back to kill/restart
        f.close()

if __name__ == "__main__":
    try:
        run_watchdog()
    except KeyboardInterrupt:
        log_msg("Watchdog stopped by user.")
        kill_all_processes()
