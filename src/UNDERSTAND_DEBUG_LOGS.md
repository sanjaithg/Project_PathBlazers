# Understanding train_td3.py Debug Logs - Complete Guide

## Document Index

You now have **3 comprehensive documents** to understand train_td3 debug logs:

### 1. **DEBUG_LOG.md** (9 KB)
**Purpose:** Technical documentation of logical flaws and fixes  
**When to Read:** Understanding the codebase improvements  
**Key Sections:**
- Root cause analysis of 6 fixed logical flaws
- Lessons learned in state management, numerical stability, code quality
- Prevention strategies for future development

---

### 2. **TRAIN_DEBUG_LOG_GUIDE.md** (14 KB) ⭐ START HERE
**Purpose:** Comprehensive reference for all log types  
**When to Read:** First time learning about train_td3 output  
**Key Sections:**
- Overview of 6 log types (timestep, episode, validation, goal, best model, shutdown)
- Field-by-field explanation of each metric
- How to interpret different log patterns
- Common pitfalls and solutions
- Real training examples with annotations
- Typical training progression timeline
- Detailed debug workflow for 3 common scenarios

**Example from Guide:**
```
Ep 2 | Reward: 325.4 | Outcome: GOAL | Goals: 1 | Collisions: 1 | Avg(10): 40.1 | Noise: 0.780

Means:
├─ Episode 2 completed
├─ Total reward: 325.4 (robot reached goal)
├─ Episode ended successfully (GOAL reached)
├─ Cumulative: 1 goal reached, 1 collision total
├─ Last 10 episodes average: 40.1 (strong improvement)
└─ Exploration noise: 0.780 (still exploring, will decrease to 0.05)
```

---

### 3. **TRAIN_LOG_QUICK_REF.md** (6.5 KB) ⭐ BOOKMARK THIS
**Purpose:** Quick reference cheat sheet for real-time monitoring  
**When to Use:** During training for quick pattern recognition  
**Key Sections:**
- One-line format decoder
- Traffic light system (🟢 GREEN = good, 🟡 YELLOW = watch, 🔴 RED = fix now)
- Real-time monitoring checklist (every 5-10 episodes)
- Quick 5-second log analysis guide
- Expected training curves at different stages
- Emergency debug commands
- Common issue/fix matrix

**Example from Quick Ref:**
```
🟢 GREEN (Everything Good):
Ep 10 | Reward: 412.1 | Outcome: GOAL | Goals: 5 | Avg(10): 250.3 | Noise: 0.680
✅ Reward > 300: Goal reached
✅ Outcome: GOAL
✅ Avg(10) > 100: Strong learning
✅ Noise decreasing: Exploration working

🔴 RED (Fix Now):
Ep 10 | Reward: -245.3 | COLLISION | Goals: 0 | Avg(10): -120.3 | Noise: 0.670
❌ Can't reach goals
❌ Hitting walls constantly
❌ Learning backward
🔧 Increase progress reward weight from 0.30 to 0.40
```

---

## Quick Navigation

**If you want to...** → **Read this:**

| Goal | Document | Section |
|------|----------|---------|
| Understand what logs mean | TRAIN_DEBUG_LOG_GUIDE | "Log Output Structure" |
| Monitor training right now | TRAIN_LOG_QUICK_REF | "Real-Time Monitoring Checklist" |
| Learn what's "normal" training | TRAIN_DEBUG_LOG_GUIDE | "Typical Training Output" |
| Identify a problem | TRAIN_LOG_QUICK_REF | "Traffic Light System" |
| Fix a specific issue | TRAIN_DEBUG_LOG_GUIDE | "Interpreting Performance Patterns" |
| Get a one-minute overview | TRAIN_LOG_QUICK_REF | "Log Output Format Cheat Sheet" |
| Deep-dive on metrics | TRAIN_DEBUG_LOG_GUIDE | "Key Metrics Tracked" |
| Understand the codebase | DEBUG_LOG | Entire document |

---

## The 6 Log Types (Quick Version)

### 1️⃣ Timestep Progress (Every 100 steps)
```
0 timesteps. Last 100 timesteps finished in 12.45 seconds
```
**Tells you:** Training speed. If this increases, you have slowdown/leak.

### 2️⃣ Episode Summary (After each episode)
```
Ep 5 | Reward: 325.4 | Outcome: GOAL | Goals: 2 | Collisions: 1 | Avg(10): 125.3 | Noise: 0.750
```
**Tells you:** How well the agent performed. Watch `Avg(10)` trend upward.

### 3️⃣ Validation Report (Every 5000 timesteps)
```
Average Reward over 10 Evaluation Episodes, Epoch 1: 45.32, Collision Rate: 0.30
```
**Tells you:** True performance on fresh goals. Should improve each epoch.

### 4️⃣ Goal Achievement (When target reached)
```
GOAL REACHED! Updating goal on next reset.
```
**Tells you:** Robot found the target. Celebrate! 🎉

### 5️⃣ Best Model Found (When beating record)
```
New best model found! Reward: 287.45. Saving...
```
**Tells you:** Policy improved. Best weights saved for deployment.

### 6️⃣ Shutdown Message (On CTRL+C)
```
Caught CTRL+C. Shutting down gracefully and saving model...
```
**Tells you:** Graceful exit, models saved.

---

## Most Important Metric: `Avg(10)`

The **rolling 10-episode average reward** is your primary progress indicator.

### Timeline Expectations
```
Ep 1:    Avg(10): -245.3  ← Random exploration (most episodes fail)
Ep 5:    Avg(10):  40.1   ← Learning starting
Ep 10:   Avg(10): 125.3   ← Good progress  ✅
Ep 20:   Avg(10): 250.3   ← Strong learning
Ep 50:   Avg(10): 350.1   ← Near convergence
Ep 100:  Avg(10): 420.5   ← Converged  ✅✅✅
```

**If your `Avg(10)` doesn't increase:**
→ Learning is stalled
→ Go to "Debug Workflow" in TRAIN_DEBUG_LOG_GUIDE
→ Most likely: Adjust reward function weights

---

## Three Scenarios You'll See

### ✅ Scenario 1: Healthy Training
```
Ep 1:  Reward: -245.3 | Outcome: COLLISION | Avg(10): -245.3 | Noise: 0.800
Ep 5:  Reward: 412.1  | Outcome: GOAL | Avg(10): 125.3 | Noise: 0.720
Ep 10: Reward: 498.3  | Outcome: GOAL | Avg(10): 250.3 | Noise: 0.680

Validating: Epoch 1: 45.32 reward, 0.30 collision rate
Validating: Epoch 2: 150.45 reward, 0.15 collision rate ✅ Improving
Validating: Epoch 3: 287.32 reward, 0.08 collision rate ✅ Converging
```
**Action:** Keep monitoring, training is working normally.

---

### ⚠️ Scenario 2: Stalled Learning
```
Ep 10: Reward: 45.3  | Avg(10): 42.1  | Noise: 0.720
Ep 11: Reward: 48.2  | Avg(10): 43.8  | Noise: 0.710
Ep 12: Reward: 41.5  | Avg(10): 44.2  | Noise: 0.700
...
[No improvement after 20+ episodes]

Validating: Epoch 5: 45.32 reward (same as Epoch 1!)
```
**Action:** 
1. Check `Avg(10)` - is it flat?
2. Are goals being reached? (search logs for "GOAL REACHED")
3. Adjust reward weights in `gazebo_env.py:get_reward()`
4. See "Scenario 1: Training Stalled" in TRAIN_DEBUG_LOG_GUIDE

---

### ❌ Scenario 3: No Goals Reached
```
Ep 1:  Reward: -245.3 | Outcome: COLLISION | Goals: 0 | Avg(10): -245.3
Ep 2:  Reward: -215.4 | Outcome: COLLISION | Goals: 0 | Avg(10): -230.4
Ep 3:  Reward: -198.7 | Outcome: TIMEOUT | Goals: 0 | Avg(10): -219.8
[Never see "GOAL REACHED" after 20+ episodes]
```
**Action:**
1. Goals count stuck at 0
2. Robot not navigating to target
3. Increase progress reward weight from 0.30 to 0.40
4. See "Scenario 2: No Goal Reaching" in TRAIN_DEBUG_LOG_GUIDE

---

## The 5-Minute Monitoring Routine

While training is running, every 5 minutes:

1. **Find latest episode line:**
   ```bash
   tail -1 <training_output> | grep "Ep"
   ```

2. **Check these fields:**
   - ✅ Is `Avg(10)` higher than 5 minutes ago?
   - ✅ Is `Outcome` mostly GOAL (not COLLISION)?
   - ✅ Is `Noise` decreasing (0.8 → 0.05)?
   - ✅ Is `Goals` count increasing?

3. **If all ✅:** Leave it running, check again in 10 minutes
4. **If any ❌:** Open TRAIN_LOG_QUICK_REF.md, find your issue pattern

---

## Real Scenario: Reading Live Logs

### Your terminal shows:
```
0 timesteps. Last 100 timesteps finished in 12.45 seconds
Ep 1 | Reward: -245.3 | Outcome: COLLISION | Goals: 0 | Collisions: 1 | Avg(10): -245.3 | Noise: 0.800
Ep 2 | Reward: 325.4 | Outcome: GOAL | Goals: 1 | Collisions: 1 | Avg(10): 40.1 | Noise: 0.780
Ep 3 | Reward: -156.2 | Outcome: TIMEOUT | Goals: 1 | Collisions: 1 | Avg(10): -25.4 | Noise: 0.760
Ep 4 | Reward: 412.1 | Outcome: GOAL | Goals: 2 | Collisions: 1 | Avg(10): 110.3 | Noise: 0.740
Ep 5 | Reward: 498.3 | Outcome: GOAL | Goals: 3 | Collisions: 1 | Avg(10): 195.2 | Noise: 0.720
```

### What this means:
1. ✅ **Ep 1-2:** Random start → learning begins (Avg: -245 → 40)
2. ✅ **Ep 3-5:** Finding pattern (Avg: 110 → 195)
3. ✅ **Noise:** Decreasing normally (0.8 → 0.72)
4. ✅ **Goals:** Increasing (0 → 3 goals in 5 episodes)
5. ✅ **Collisions:** Not growing (stuck at 1)

**Verdict:** 🟢 GREEN - Training healthy, keep monitoring

---

## When to Adjust Reward Function

| Symptom | Issue | Fix | File |
|---------|-------|-----|------|
| `Avg(10)` flat for 20+ ep | Rewards too balanced | `↑ R_progress from 0.30 to 0.40` | gazebo_env.py:558 |
| `Goals: 0` | Can't reach target | `↑ R_progress or ↓ R_barrier` | gazebo_env.py:500+ |
| `Collisions` rapid | Hits walls | `↑ penalty cap from -2000 to -2500` | gazebo_env.py:500 |
| `Avg(10)` oscillates | Too much noise | `↓ expl_decay_steps` | td3_config.yaml |

All reward changes in: [gazebo_env.py](bots/bots/td3_rl/gazebo_env.py)

---

## Pro Tips

### 💡 Tip 1: Grep for patterns
```bash
# Find all goal achievements
grep "GOAL REACHED" training.log | wc -l

# Find collision count
grep "Outcome: COLLISION" training.log | wc -l

# Check validation trend
grep "Average Reward" training.log | tail -10
```

### 💡 Tip 2: Plot results
```python
import numpy as np; import matplotlib.pyplot as plt
evals = np.load("bots/bots/td3_rl/results/TD3_Mecanum.npy")
plt.plot(evals); plt.xlabel("Epoch"); plt.ylabel("Reward"); plt.show()
```

### 💡 Tip 3: Use keyboard shortcuts
- Ctrl+F in terminal: Search "GOAL REACHED"
- Ctrl+A + Ctrl+C: Copy all logs
- | tail -50: See last 50 lines

---

## Summary

**You now have:**
- ✅ Complete understanding of 6 log types
- ✅ Real-time monitoring patterns
- ✅ Diagnostic checklist for common issues
- ✅ Expected training trajectories
- ✅ Debug commands for quick fixes

**Next Step:**
1. Start training: `ros2 launch bots train_td3.launch.py`
2. Bookmark TRAIN_LOG_QUICK_REF.md
3. Monitor `Avg(10)` trending upward
4. Use DEBUG_LOG_GUIDE.md when stuck
5. Adjust reward weights if needed

**Good luck! 🚀**

---

## Document Locations

```
/home/dark/ros2_ws/src2/Project_PathBlazers/src/
├── DEBUG_LOG.md                    # Technical fixes & lessons
├── TRAIN_DEBUG_LOG_GUIDE.md        # Comprehensive reference ⭐
├── TRAIN_LOG_QUICK_REF.md          # Quick cheat sheet ⭐
├── GETTING_STARTED.md              # Project setup
├── BOT_WORLD_SPECS.md              # Robot specifications
├── IMPROVEMENTS_SUMMARY.md         # Latest improvements
└── README.md                       # Project overview
```

All guides are markdown files you can read in VS Code or any text editor.
