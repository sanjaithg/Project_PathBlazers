# Debugging & Tuning Guide

## Issue 1: High Collision Rate in Early Episodes

### Symptoms:
```
Ep 1-5: Mostly collisions (-200 to -500 rewards)
Goal success rate: < 30%
```

### Root Causes & Solutions:

**Cause A: Obstacle Avoidance Reward Too Weak**
```python
# WEAK (current):
R_obstacle = -((SAFE - min_laser) / SAFE) ** 1.5 * 50

# STRONGER:
R_obstacle = -((SAFE - min_laser) / SAFE) ** 1.5 * 100  # Double weight

# Or adjust weights:
0.25 * R_obstacle → 0.35 * R_obstacle  (increase proportion)
```

**Cause B: LIDAR Data Missing**
```bash
# Check if LIDAR is publishing:
ros2 topic list | grep scan
ros2 topic echo /scan | head -20

# If no data: Check gazebo plugin in .gazebo file
# Verify sensor in URDF: <sensor name="lidar" type="ray">
```

**Cause C: Collision Threshold Too High**
```python
# Current thresholds:
GOAL_REACHED_DIST = 0.4
COLLISION_DIST = 0.50

# Try:
COLLISION_DIST = 0.45  # More conservative
```

### Quick Fix (Progressive):
1. Reduce progress weight: 0.40 → 0.30
2. Increase obstacle weight: 0.25 → 0.40
3. Rebuild and retrain 5-10 episodes
4. Adjust if needed

---

## Issue 2: Robot Spinning in Circles

### Symptoms:
```
Ep X: High negative rewards (< -100)
Robot rotates without moving forward
Outcome: TIMEOUT (stuck)
```

### Root Causes & Solutions:

**Cause A: Heading Reward Too Strong**
```python
# Current:
R_heading = np.cos(theta) * 5.0

# Reduce to:
R_heading = np.cos(theta) * 2.0  # Gentler guidance
```

**Cause B: Angular Penalty Ineffective**
```python
# If spinning still happens:
penalty_angular = abs(action[2]) * 3.0

# Increase to:
penalty_angular = abs(action[2]) * 5.0  # Stronger angular suppression

# Or increase its weight:
0.05 * (-penalty_angular) → 0.10 * (-penalty_angular)
```

**Cause C: State Space Missing Rotation Context**
```python
# Ensure sin/cos encoding is used:
sin_theta = np.sin(theta)  # ✓ Should have this
cos_theta = np.cos(theta)  # ✓ Should have this

# NOT raw theta:
robot_state = [distance, sin_theta, cos_theta, ...]  # ✓ Correct
robot_state = [distance, theta, ...]                  # ✗ Wrong
```

### Progressive Fix:
1. Monitor `Noise` value in output - if high, reduce angular actions
2. Check if theta encoding is sin/cos (not raw)
3. Reduce angular penalty weight gradually

---

## Issue 3: Robot Moves Backward (Wrong Direction)

### Symptoms:
```
Robot moves away from goal
Reward: Large negative
```

### Root Causes & Solutions:

**Cause A: Action Scaling Incorrect**
```python
# Check in spawn_robot.launch.py:
vel_cmd.linear.x = float(action[0])  # -1 to +1, should be: -1.0 to +1.0 m/s
vel_cmd.linear.y = float(action[1])  # Mecanum sideways

# Verify robot accepts Twist messages:
ros2 topic echo /cmd_vel | head -10
```

**Cause B: Odometry Flipped**
```python
# Check gazebo_env.py odom_callback:
self.odom_x = float(self.last_odom.pose.pose.position.x)  # ✓ Correct
self.odom_y = float(self.last_odom.pose.pose.position.y)

# Verify odometry direction:
ros2 topic echo /odometry/filtered | head -20

# Check if robot moves forward when sending positive x:
ros2 topic pub -1 /cmd_vel geometry_msgs/Twist "{linear: {x: 0.5, y: 0.0, z: 0.0}}"
```

**Cause C: Distance Calculation Wrong**
```python
# Verify distance to goal calculation:
distance = np.linalg.norm([self.odom_x - self.goal_x, self.odom_y - self.goal_y])

# Should decrease as robot moves toward goal
```

---

## Issue 4: Training Stuck (No Improvement)

### Symptoms:
```
Episodes 20-50: Reward flat or decreasing
Success rate plateaus at low value
```

### Root Causes & Solutions:

**Cause A: Learning Rate Too High**
```python
# In train_td3.py, check:
self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
# Default lr=0.001, try:
self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0005)
self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.0005)
```

**Cause B: Batch Size Too Large**
```yaml
# In td3_config.yaml:
batch_size: 128  # Try reducing to:
batch_size: 64   # Faster, noisier updates
```

**Cause C: Neural Network Too Small**
```python
# In train_td3.py Actor/Critic:
self.layer_1 = nn.Linear(state_dim, 800)   # Try:
self.layer_1 = nn.Linear(state_dim, 512)
self.layer_2 = nn.Linear(800, 600)         # Try:
self.layer_2 = nn.Linear(512, 256)
```

**Cause D: Exploration Too Low**
```yaml
expl_noise: 1.0  # Currently should be okay
expl_min: 0.1    # Minimum exploration
# If stuck, try higher initial:
expl_noise: 1.5
expl_min: 0.2
```

### Debugging Steps:
```bash
# 1. Reset training:
rm -f bots/bots/td3_rl/runs/*
rm -f TD3_Mecanum*

# 2. Start fresh with reduced complexity:
# - Reduce max_ep to 50 (quick test)
# - Monitor first 10 episodes carefully
# - Check if any episodes reach goal

# 3. If some reach goal:
# - Increase max_ep gradually
# - Tune reward weights

# 4. If none reach goal:
# - Issue is with obstacle avoidance
# - Increase obstacle reward weight
# - Reduce progress weight
```

---

## Issue 5: GPU Not Available

### Symptoms:
```
2026-01-03 19:37:00: Could not find cuda drivers
device: cpu
```

### This is Normal!
```python
# Training on CPU is fine for small networks
# Expected timing:
CPU:  1-2 hours for 150 episodes
GPU:  15-30 min for 150 episodes

# To improve CPU speed:
1. Reduce batch_size: 128 → 64
2. Reduce network size: 800 → 512
3. Reduce evaluation frequency: eval_freq: 500 → 1000
```

---

## Real-Time Monitoring Checklist

### Every 5 Episodes:
- [ ] Check Avg(10) is increasing
- [ ] Verify Collisions < 1 per episode
- [ ] Ensure Goals > 0 (at least 1 per 5 episodes)
- [ ] Monitor Noise decreasing (should decay)

### Every 25 Episodes:
- [ ] Compare Ep1-25 vs Ep26-50 average rewards
- [ ] Should see improvement of at least +50 reward
- [ ] Collision rate should be decreasing
- [ ] Save model output (check terminal)

### Every 50 Episodes:
- [ ] Full evaluation: 10 consecutive episodes
- [ ] Success rate should reach 60%+
- [ ] Check for mode collapse (all same behavior)
- [ ] Consider hyperparameter adjustment if stuck

---

## Reward Function Tuning Template

Start with these baseline weights:
```python
# Baseline (current):
w_progress = 0.40
w_obstacle = 0.25
w_smoothness = 0.10
w_heading = 0.10
w_time = 0.10
w_angular = 0.05

# If too many collisions:
w_progress = 0.30  ↓
w_obstacle = 0.40  ↑ (more safety focus)

# If spinning in circles:
w_heading = 0.05   ↓
w_angular = 0.10   ↑ (more rotation suppression)

# If moving too slowly:
w_time = 0.15      ↑
w_smoothness = 0.05 ↓ (allow jittery but fast)

# Always verify: sum(weights) ≈ 1.0
```

---

## TensorBoard Analysis

### What Each Graph Means:

```
1. episode_reward → Total reward per episode
   ✓ Should trend upward over time
   ✓ Expect: 0-50 → 200-400 → 400-500

2. avg_reward → Rolling average (smoothed)
   ✓ Shows learning trend clearly
   ✓ Target: >300 by episode 100

3. success_rate → % episodes reaching goal
   ✓ Should increase monotonically
   ✓ Target: >80% by episode 100

4. collision_count → Crashes per episode
   ✓ Should decrease over time
   ✓ Target: <1 per 10 episodes by episode 150

5. min_lidar_distance → Minimum obstacle distance
   ✓ Should stay stable or increase
   ✓ Target: >0.5m (no collisions)

6. Reward components breakdown:
   ✓ R_progress should dominate by episode 150
   ✓ R_obstacle should stabilize (fewer near-miss)
   ✓ R_smoothness should improve (less jerky)
```

---

## Performance Tuning for Speed

### If Training is Too Slow:

```python
# 1. Reduce environment_dim:
config: environment_dim: 20  → 10
# Faster LIDAR processing, less state

# 2. Reduce eval_freq:
config: eval_freq: 500  → 1000
# Evaluate less often

# 3. Smaller network:
layer_1: 800 → 512
layer_2: 600 → 256

# 4. Reduce physics accuracy:
# (only if not in training)
# Gazebo real_time_factor: 2 → 3

# 5. Disable TensorBoard:
# Comment: self.writer.add_scalar(...)
# Log only every 10 episodes instead
```

### CPU Optimization:
```bash
# Set process priority (if allowed):
nice -n -5 ros2 run bots train_td3

# Or use taskset to bind to specific cores:
taskset -c 0-3 ros2 run bots train_td3
```

---

## Common Error Messages

### "No LaserScan data"
```
FIX: Wait 5 seconds before starting training
     Gazebo takes time to publish /scan
```

### "SetEntityPose service not available"
```
FIX: Ensure my_launch_file.py is running
     Check that gz_sim bridge is active
     Verify Gazebo server status
```

### "Action vector must be size 3"
```
FIX: Check action_dim = 3 in train_td3.py
     Verify Actor outputs 3 values
     Check step() function receives correct shape
```

### "Negative tensor in log"
```
FIX: This is numerical stability issue
     Check: min_laser is being clipped correctly
     Ensure: all rewards are finite (not inf/-inf)
```

---

## Final Checklist Before Extended Training

- [ ] Syntax verification: `python3 -m py_compile *.py`
- [ ] First 10 episodes run without crashes
- [ ] At least 1 goal reached in first 10 episodes
- [ ] LIDAR data visible and changing
- [ ] Odometry updates correctly
- [ ] Model file path is writable
- [ ] GPU status checked (or CPU OK for 2-3 hours)
- [ ] Reward trends make sense (increasing)
- [ ] TensorBoard accessible for monitoring

**Ready to train!** 🚀
