# Train_TD3 Debug Log Logic Guide

## Overview

The `train_td3.py` script produces comprehensive debug logs during training. This guide explains:
1. **What logs are produced**
2. **What each metric means**
3. **How to interpret the output**
4. **How to troubleshoot based on logs**

---

## Log Output Structure

### 1. **Timestep Progress Logs** (Every 100 timesteps)
```
{timestep} timesteps. Last 100 timesteps finished in {time:.2f} seconds
```

**Location:** Line 444  
**Frequency:** Every 100 steps  
**Purpose:** Track training speed

**Example:**
```
0 timesteps. Last 100 timesteps finished in 12.45 seconds
100 timesteps. Last 100 timesteps finished in 15.32 seconds
200 timesteps. Last 100 timesteps finished in 14.87 seconds
```

**Interpretation:**
- Time should be relatively consistent (12-20 seconds typical)
- If time increases significantly: memory leak or Gazebo slowdown
- If time decreases: efficiency improvement (rare)

---

### 2. **Episode Summary Logs** (After each episode)
```
Ep {episode_num} | Reward: {reward:.1f} | Outcome: {outcome} | Goals: {total_goals} | Collisions: {total_collisions} | Avg(10): {avg_recent:.1f} | Noise: {noise:.3f}
```

**Location:** Line 505-509  
**Frequency:** After each episode completes  
**Purpose:** Track per-episode performance

**Example:**
```
Ep 1 | Reward: -245.3 | Outcome: COLLISION | Goals: 0 | Collisions: 1 | Avg(10): -245.3 | Noise: 0.800
Ep 2 | Reward: 325.4 | Outcome: GOAL | Goals: 1 | Collisions: 1 | Avg(10): 40.1 | Noise: 0.780
Ep 3 | Reward: -156.2 | Outcome: TIMEOUT | Goals: 1 | Collisions: 1 | Avg(10): -25.4 | Noise: 0.760
Ep 4 | Reward: 412.1 | Outcome: GOAL | Goals: 2 | Collisions: 1 | Avg(10): 110.3 | Noise: 0.740
```

**Fields Explained:**

| Field | Meaning | Good Values | Bad Values |
|-------|---------|-------------|-----------|
| `Ep N` | Episode number | Sequential | - |
| `Reward: X` | Total reward for episode | +300 to +500 (goal) | -500 to -200 (collision) |
| `Outcome` | How episode ended | GOAL | COLLISION, TIMEOUT |
| `Goals: N` | Cumulative goals reached | Increasing | Stuck at 0-2 |
| `Collisions: N` | Cumulative collisions | Low growth | Rapid growth |
| `Avg(10): X` | Average reward last 10 ep | Trending up | Trending down |
| `Noise: X` | Exploration noise level | Decreasing 0.8→0.05 | Not changing |

**Outcome Meanings:**
- **GOAL**: Robot reached target successfully (reward > 0 typically)
- **COLLISION**: Barrier reward heavily penalized agent (reward < -100)
- **TIMEOUT**: Episode max steps reached (reward varies)

---

### 3. **Validation Logs** (Every eval_freq timesteps, default 5000)
```
Validating at timestep {timestep}
..............................................
Average Reward over {eval_episodes} Evaluation Episodes, Epoch {epoch}: {avg_reward:.2f}, Collision Rate: {avg_col:.2f}
..............................................
```

**Location:** Line 453-478  
**Frequency:** Every 5000 timesteps (configurable)  
**Purpose:** Evaluate policy performance on fresh goals

**Example:**
```
Validating at timestep 5000
..............................................
Average Reward over 10 Evaluation Episodes, Epoch 1: 45.32, Collision Rate: 0.30
..............................................
```

**Interpretation:**
- **Avg Reward:** Should trend upward over time (45 → 150 → 300+)
- **Collision Rate:** Should trend downward (0.5 → 0.3 → 0.1)
- If stalled: learning plateau, reward scaling issue, or exploration too aggressive

---

### 4. **Best Model Found Log**
```
New best model found! Reward: {avg_reward:.2f}. Saving...
```

**Location:** Line 471  
**Frequency:** When validation reward exceeds previous best  
**Purpose:** Track performance breakthroughs

**Example:**
```
New best model found! Reward: 52.45. Saving...
New best model found! Reward: 187.32. Saving...
New best model found! Reward: 312.10. Saving...
```

**What This Means:**
- Model is improving
- Previous best weights saved as `{filename}_best.pth`
- Use this model for deployment if training is interrupted

---

### 5. **Goal Reached Logs**
```
GOAL REACHED! Updating goal on next reset.
```

**Location:** Line 600  
**Frequency:** When target is reached mid-episode  
**Purpose:** Track individual goal completions

**Example:**
```
Ep 2 | Reward: 325.4 | Outcome: GOAL | ...
GOAL REACHED! Updating goal on next reset.
Ep 3 | Reward: 412.1 | Outcome: GOAL | ...
GOAL REACHED! Updating goal on next reset.
```

---

### 6. **Shutdown Logs**
```
Caught CTRL+C. Shutting down gracefully and saving model...
Model and evaluation results saved at timestep {timestep}.
Robot stopped (cmd_vel = 0)
```

**Location:** Lines 606-618  
**Frequency:** On CTRL+C interruption  
**Purpose:** Graceful cleanup

**Saved Files on Interrupt:**
- `{filename}_interrupted_actor.pth` - Latest actor network
- `{filename}_interrupted_critic.pth` - Latest critic network
- `{filename}_interrupted.npy` - Evaluation history
- `odometry_history.txt` - Robot movement log (if enabled)

---

## Log Hierarchy & Typical Training Output

### **Phase 1: Initialization (Timestep 0-100)**
```
0 timesteps. Last 100 timesteps finished in 12.45 seconds
Loaded Parameters:
seed: 0
eval_freq: 500
max_ep: 500
...
Ep 1 | Reward: -245.3 | Outcome: COLLISION | Goals: 0 | Collisions: 1 | Avg(10): -245.3 | Noise: 0.800
```

### **Phase 2: Early Training (Timestep 100-5000)**
```
100 timesteps. Last 100 timesteps finished in 15.32 seconds
Ep 2 | Reward: 325.4 | Outcome: GOAL | Goals: 1 | Collisions: 1 | Avg(10): 40.1 | Noise: 0.780
Ep 3 | Reward: -156.2 | Outcome: TIMEOUT | Goals: 1 | Collisions: 1 | Avg(10): -25.4 | Noise: 0.760
GOAL REACHED! Updating goal on next reset.
...
```

### **Phase 3: Validation (Timestep 5000+)**
```
Validating at timestep 5000
..............................................
Average Reward over 10 Evaluation Episodes, Epoch 1: 45.32, Collision Rate: 0.30
..............................................
New best model found! Reward: 45.32. Saving...

Ep 25 | Reward: 412.1 | Outcome: GOAL | Goals: 5 | Collisions: 8 | Avg(10): 156.3 | Noise: 0.650
```

### **Phase 4: Advanced Training (Timestep 50000+)**
```
Validating at timestep 50000
..............................................
Average Reward over 10 Evaluation Episodes, Epoch 10: 287.45, Collision Rate: 0.10
..............................................
New best model found! Reward: 287.45. Saving...

Ep 200 | Reward: 498.3 | Outcome: GOAL | Goals: 85 | Collisions: 32 | Avg(10): 425.1 | Noise: 0.150
```

---

## Key Metrics Tracked

### Per-Episode Tracking
```python
self.total_goals_reached = 0          # Cumulative goal count
self.total_collisions = 0             # Cumulative collision count
self.recent_rewards = []              # Last 10 episode rewards
```

### Variables Monitored
```python
self.best_avg_reward = -np.inf        # Best validation reward seen
self.expl_noise = 0.8                 # Exploration noise (decays)
episode_reward                        # Accumulates per-step rewards
episode_timesteps                     # Steps in current episode
```

### Validation Metrics
```python
avg_reward_eval                       # Mean reward over eval_episodes
avg_col_eval                          # Collision rate (collisions / eval_episodes)
```

---

## Interpreting Performance Patterns

### ✅ **Healthy Training Signs**
```
Ep 1 | Reward: -245.3 | Outcome: COLLISION | Goals: 0 | Avg(10): -245.3 | Noise: 0.800
Ep 2 | Reward: 325.4  | Outcome: GOAL | Goals: 1 | Avg(10): 40.1 | Noise: 0.780
Ep 3 | Reward: 412.1  | Outcome: GOAL | Goals: 2 | Avg(10): 125.3 | Noise: 0.760
Validating: Epoch 1: 45.32 reward, 0.30 collision rate
Validating: Epoch 2: 150.45 reward, 0.15 collision rate ✅ Improving
Validating: Epoch 3: 287.32 reward, 0.08 collision rate ✅ Better
```

**Indicators:**
- ✅ Avg(10) trending upward
- ✅ Collision rate decreasing
- ✅ Goals reached increasing
- ✅ Validation rewards improving each epoch
- ✅ Model saves increasing (best reward found)

---

### ⚠️ **Warning Signs**

#### **1. Stuck Learning (Flat Rewards)**
```
Ep 10 | Reward: 45.3  | Outcome: GOAL | Goals: 3 | Avg(10): 42.1 | Noise: 0.720
Ep 11 | Reward: 48.2  | Outcome: GOAL | Goals: 4 | Avg(10): 43.8 | Noise: 0.710
Ep 12 | Reward: 41.5  | Outcome: GOAL | Goals: 5 | Avg(10): 44.2 | Noise: 0.700
...
Validating: Epoch 5: 45.32 reward, 0.30 collision rate (same as epoch 1)
```

**Causes:**
- Learning rate too small
- Reward function scaling issue
- Exploration noise too low
- Network capacity insufficient

**Fix:** Check `td3_config.yaml` for learning rate, increase exploration noise

---

#### **2. Oscillating Rewards (High Variance)**
```
Ep 5 | Reward: 412.1  | Outcome: GOAL | Goals: 2 | Avg(10): 150.3 | Noise: 0.750
Ep 6 | Reward: -315.2 | Outcome: COLLISION | Goals: 2 | Avg(10): 42.1 | Noise: 0.740
Ep 7 | Reward: 498.3  | Outcome: GOAL | Goals: 3 | Avg(10): 95.1 | Noise: 0.730
Ep 8 | Reward: -245.6 | Outcome: COLLISION | Goals: 3 | Avg(10): 37.3 | Noise: 0.720
```

**Causes:**
- Exploration noise too high
- Reward function too aggressive/noisy
- Barrier reward oscillating

**Fix:** Check barrier function clipping, reduce noise decay rate

---

#### **3. No Goal Reaching**
```
Ep 1 | Reward: -245.3 | Outcome: COLLISION | Goals: 0 | Avg(10): -245.3 | Noise: 0.800
Ep 2 | Reward: -215.4 | Outcome: COLLISION | Goals: 0 | Avg(10): -230.4 | Noise: 0.780
Ep 3 | Reward: -198.7 | Outcome: TIMEOUT | Goals: 0 | Avg(10): -219.8 | Noise: 0.760
...
(no "GOAL REACHED" logs after 20+ episodes)
```

**Causes:**
- Progress reward too weak
- Collision barrier too aggressive
- State space issue (e.g., goal distance not updating)

**Fix:** Check reward function weights in `get_reward()`, verify goal updates

---

#### **4. Collision Rate Not Decreasing**
```
Validating: Epoch 1: 45.32 reward, 0.50 collision rate
Validating: Epoch 2: 78.45 reward, 0.45 collision rate
Validating: Epoch 3: 125.32 reward, 0.48 collision rate ⚠️ Increased!
```

**Causes:**
- Barrier function penalty not strong enough
- LIDAR range too short
- Robot gets "stuck" and can't recover

**Fix:** Check barrier reward scaling, increase wall avoidance weight

---

## Log Files Generated

### During Training
```
bots/bots/td3_rl/
├── pytorch_models/
│   ├── TD3_Mecanum_actor.pth           # Latest actor weights
│   ├── TD3_Mecanum_critic.pth          # Latest critic weights
│   ├── TD3_Mecanum_best_actor.pth      # Best validation actor
│   └── TD3_Mecanum_best_critic.pth     # Best validation critic
└── results/
    └── TD3_Mecanum.npy                 # Array of validation rewards
```

### Analysis
```python
import numpy as np
import matplotlib.pyplot as plt

# Load evaluation history
evaluations = np.load("bots/bots/td3_rl/results/TD3_Mecanum.npy")

# Plot training progress
plt.plot(evaluations)
plt.xlabel("Validation Epoch")
plt.ylabel("Average Reward")
plt.title("Training Progress")
plt.grid(True)
plt.savefig("training_progress.png")
```

---

## Using `analyze_training.py`

The project includes a utility to visualize results:

```bash
cd /home/dark/ros2_ws/src2/Project_PathBlazers/src
python3 bots/bots/td3_rl/analyze_training.py
```

**Output:**
```
Loading results from: .../results/TD3_Mecanum.npy

--- Training Analysis ---
Total Evaluations: 15
Max Reward: 312.45
Min Reward: 23.32
Average Reward (All): 168.90
Average Reward (Last 10): 287.45

Good Epochs (> 0): 14
Bad Epochs (< -150): 1

Plot saved to: .../results/TD3_Mecanum_plot.png
```

---

## Common Debug Workflow

### **Scenario 1: Training Stalled at Low Reward**
```
Validating: Epoch 5: 45.32 reward (same as Epoch 1)
```

**Debug Steps:**
1. Check `Avg(10)` in episode logs - is it flat?
2. Are goals being reached at all? (search for "GOAL REACHED")
3. Check noise level - is it stuck at 0.050 (minimum)?
4. Review `gazebo_env.get_reward()` - is progress reward too weak?

**Fix Example:**
```python
# In get_reward(), increase progress weight from 0.30 to 0.40
total_reward = (
    0.40 * R_progress_pbrs +  # Increased from 0.30
    0.20 * R_velocity +
    0.15 * R_barrier_adaptive +  # Decreased from 0.20
    ...
)
```

---

### **Scenario 2: High Collision Rate**
```
Validating: Epoch 3: Collision Rate: 0.45
```

**Debug Steps:**
1. Check `Collisions:` count in episode logs - growing too fast?
2. Review barrier reward logs in `gazebo_env.get_barrier_reward()`
3. Is LIDAR range sufficient? (Check `max_distance = 3.5m`)
4. Is deviation counter resetting properly?

**Fix Example:**
```python
# In get_barrier_reward(), strengthen penalty
denom = max(dist - phys_limit, 1e-3)  # Increase from 1e-4
penalty = -1.0 / denom
penalty = max(penalty, -2500.0)  # Increase cap from -2000
```

---

## Summary: Reading the Logs

| What You See | What It Means | What To Do |
|--------------|--------------|-----------|
| `Reward: 450+, GOAL` | Successful episode | ✅ Training working |
| `Reward: -250, COLLISION` | Hit wall | ⚠️ Check barrier |
| `Avg(10)` trending up | Learning improving | ✅ Good |
| `Avg(10)` flat for 20+ eps | Stuck learning | ⚠️ Debug reward |
| `Goals: 0` after 10 eps | Can't reach goals | ⚠️ Progress reward weak |
| `Collisions` growing fast | Poor obstacle avoidance | ⚠️ Increase barrier |
| `Validating... New best` | New record! | ✅ Model improving |
| Noise → 0.050 | Exploration decay done | ✅ Exploitation mode |
| Time per 100 steps increasing | Memory leak/slowdown | ⚠️ Check system |

---

## Next Steps

1. **Monitor first 10 episodes** - Should see at least 1-2 successful goals by Ep 5
2. **Check validation epoch 1** - Should have 20-50+ reward by 5000 timesteps
3. **Watch Avg(10)** - Should increase by ~50-100 per epoch for first 10 epochs
4. **Use analyze_training.py** - Plot results to visualize convergence
5. **Save best model** - Use `TD3_Mecanum_best` weights for deployment
