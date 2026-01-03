# Quick Start Guide - TD3 Navigation Training

## Setup (One Time)
```bash
cd /home/dark/ros2_ws/src2/Project_PathBlazers/src
colcon build --packages-select bots navigator
source install/setup.bash
```

## Training Start (Terminal 1 - Gazebo & ROS)
```bash
cd /home/dark/ros2_ws/src2/Project_PathBlazers/src
source install/setup.bash
ros2 launch bots train_td3.launch.py
```

## Training Start (Terminal 2 - TD3 Agent)
```bash
cd /home/dark/ros2_ws/src2/Project_PathBlazers/src
source install/setup.bash
ros2 run bots train_td3
```

---

## Training Monitoring

### Real-Time Metrics (in Terminal 2):
- **Ep X | Reward: Y | Outcome: GOAL|COLLISION|TIMEOUT**
  - Outcome legend:
    - GOAL: Reached target successfully (+500 reward)
    - COLLISION: Hit obstacle (-500 reward)
    - TIMEOUT: Stuck for too long (-200 reward)
  
- **Goals: X | Collisions: Y** - Cumulative episode statistics
- **Avg(10): Z** - Average reward over last 10 episodes (target: >300)
- **Noise: N** - Exploration noise level (gradually decays)

### TensorBoard Visualization:
```bash
# In Terminal 3:
tensorboard --logdir=/home/dark/ros2_ws/src2/Project_PathBlazers/src/bots/bots/td3_rl/runs

# Then open: http://localhost:6006
```

---

## Key Training Phases

### Phase 1: Collision Learning (Episodes 1-50)
- Robot learns basic obstacle avoidance
- Success rate: 25-60%
- Expected: Some crashes, but fewer each episode

### Phase 2: Goal Navigation (Episodes 51-150)
- Robot balances progress and safety
- Success rate: 60-85%
- Expected: Smooth navigation, occasional crashes

### Phase 3: Optimization (Episodes 151+)
- Fine-tuning motion smoothness
- Success rate: 85%+
- Expected: Mostly smooth, efficient paths

---

## New Reward Function (Improved)

**Components:**
1. **Progress Reward** (40%): Moving closer to goal
2. **Obstacle Avoidance** (25%): Maintaining distance from obstacles
3. **Smoothness** (10%): Avoiding jerky movements
4. **Heading Guidance** (10%): Gentle direction correction
5. **Time Efficiency** (10%): Rewarding forward motion
6. **Angular Restraint** (5%): Reducing unnecessary rotation

**Special Rewards:**
- Goal Reached: +500
- Collision: -500
- Timeout: -200

---

## New State Space (29 dimensions)

### LIDAR (20 dims):
- Sampled laser scan rays

### Robot State (9 dims):
- distance_to_goal
- sin(angle_to_goal)
- cos(angle_to_goal)
- linear_x_velocity
- linear_y_velocity
- angular_velocity
- front_obstacle_proximity
- left_obstacle_proximity
- right_obstacle_proximity

**Why these changes?**
- sin/cos encoding for circular continuity
- Sector-based obstacle awareness
- Better state representation for NN

---

## Troubleshooting

### Problem: All episodes are collisions
**Solution**: 
- Increase obstacle avoidance weight in reward
- Reduce exploration noise (expl_noise)
- Check LIDAR data is publishing

### Problem: Robot reaches goal but keeps going
**Solution**:
- Increase GOAL_REACHED_DIST from 0.4 to 0.5
- Verify goal detection in logs

### Problem: Training stuck (no improvement)
**Solution**:
- Reduce batch_size from 128 to 64
- Increase exploration noise temporarily
- Verify environment is spawning obstacles correctly

### Problem: Slow training on CPU
**Solution**:
- Reduce batch_size to 64
- Reduce max_ep from 150 to 50 for quick validation
- Use smaller neural network (reduce 800/600 to 512/256)

---

## Success Indicators ✓

- [x] Phase 1 complete: 5+ consecutive goal-reaches
- [x] Phase 2 complete: 80% success rate sustained
- [x] Phase 3 complete: Smooth paths with <2% collision
- [x] Ready for real robot: Model saved to `TD3_Mecanum.pth`

---

## Save & Load Models

### Load Pre-trained Model:
```bash
# In config/td3_config.yaml:
load_model: true  # Set to true
file_name: "TD3_Mecanum"  # Load saved model
```

### Inference (Testing):
```bash
ros2 run bots test_td3
```

---

## Next Steps (Dynamic Obstacles)

When ready for moving obstacles:
1. Update world.sdf to add moving actors
2. Add velocity to state space
3. Implement velocity prediction in reward
4. Retrain with moving targets (50-100 episodes)

---

## Notes

- Training uses sim_time (simulated time)
- Gazebo physics runs at 1000 Hz
- Each step advances 0.1 seconds
- Robot action applied, then 0.1s simulation, then pause
- LIDAR updates once per pause cycle
