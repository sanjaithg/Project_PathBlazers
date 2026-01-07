# Getting Started

## Quick Setup

### 1. Build the Project
```bash
cd /home/dark/ros2_ws/src2/Project_PathBlazers
colcon build
source install/setup.bash
```

### 2. Launch Training
```bash
ros2 launch bots train_td3.launch.py
```

Training will start immediately with the TD3 algorithm and enhanced reward function.

---

## Project Structure

```
bots/                          # Main ROS2 package
├── bots/td3_rl/
│   ├── gazebo_env.py         # Environment with shape-aware rewards
│   ├── train_td3.py          # TD3 training loop
│   └── td3_config.yaml       # Configuration parameters
├── launch/
│   └── train_td3.launch.py   # Training launcher
└── urdf/
    └── mogi_bot_mecanum.urdf # Robot definition

navigator/                     # Navigation package
config/                        # Configuration files
```

---

## Key Components

### Robot: Mogi Bot Mecanum
- **Chassis:** 0.4m × 0.2m × 0.1m
- **Wheels:** Mecanum drive (omnidirectional)
- **Lidar:** 720-point 360° scan, 30 Hz
- **Action Space:** 4D (wheel velocities)
- **Observation Space:** 29D (LIDAR + goal state + sector awareness)

### Training Algorithm: TD3 (Twin Delayed DDPG)
- **Exploration Noise:** 0.8
- **Replay Buffer:** 1,000,000 steps
- **Network Architecture:** 2 hidden layers (256 units)
- **Update Frequency:** Every 2 steps

### Reward Function (Shape-Aware)
```
R_total = 0.40*R_progress + 0.25*R_obstacle + 0.10*R_smoothness 
        + 0.10*R_heading + 0.10*R_time + 0.05*R_angular

Obstacle Avoidance (CBF-inspired):
- Front/Back: Penalty at 0.45m with reciprocal barrier
- Sides: Penalty at 0.25m with reciprocal barrier
- Max penalty: -2000 for wall collision
```

---

## Expected Training Outcomes

| Phase | Episodes | Success Rate | Time |
|-------|----------|--------------|------|
| Phase 1 | 1-50 | 60% | 40-50 min |
| Phase 2 | 51-150 | 90% | 80-100 min |
| Phase 3 | 151+ | 95%+ | 40-50 min |

**Total:** ~2.5 hours on CPU for convergence

---

## Documentation

- **[BOT_WORLD_SPECS.md](BOT_WORLD_SPECS.md)** - Technical specifications and geometry
- **[IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)** - Latest improvements (2026-01-07)
- **[README.md](README.md)** - Project overview and architecture

---

## Troubleshooting

**Build fails:**
```bash
rm -rf build install log
colcon build
```

**Gazebo doesn't launch:**
```bash
gazebo --verbose
```

**Training hangs:**
- Check `/tmp` for zombie processes
- Ensure Gazebo is responding in another terminal with `gz topic list`

---

## Training Monitoring & Checkpointing

### Real-Time Logging
Training automatically logs detailed data to CSV:
```
bots/bots/td3_rl/results/training_log.csv
```

**Logged data includes:**
- Per-timestep reward components (progress, barrier, velocity, heading, etc.)
- Control commands (cmd_vel_x, cmd_vel_y, cmd_vel_z)
- Robot pose (x, y, yaw) and goal position
- Distance to goal, minimum laser distance

### Live Visualization
Run in a separate terminal while training:
```bash
python3 src/bots/bots/td3_rl/visualize_live.py
```

This displays real-time plots of:
- Total reward with rolling average
- Individual reward components
- Distance to goal over time
- Minimum laser distance (safety monitoring)
- Control commands history
- Robot XY trajectory

### Checkpointing & Resumable Training
Training automatically saves checkpoints on interruption (Ctrl+C) or completion:
```
bots/bots/td3_rl/pytorch_models/TD3_Mecanum_checkpoint.pth  # Model + optimizer states + metadata
bots/bots/td3_rl/pytorch_models/TD3_Mecanum_replay.pkl      # Replay buffer
```

**To resume training from a checkpoint:**
```bash
ros2 run bots train_td3 --ros-args -p load_model:=True
```

The training will continue from the exact timestep, episode, and exploration noise state where it was interrupted.

---

## Next Steps

1. Launch training: `ros2 launch bots train_td3.launch.py`
2. In another terminal: `ros2 run bots train_td3`
3. Monitor live: `python3 src/bots/bots/td3_rl/visualize_live.py`
4. Interrupt with Ctrl+C to save checkpoint
5. Resume later with `load_model:=True`

