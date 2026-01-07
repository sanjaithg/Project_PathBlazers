# Train_TD3 Log Quick Reference Card

## Log Output Format Cheat Sheet

### Episode Log Format
```
Ep {N} | Reward: {R} | Outcome: {O} | Goals: {G} | Collisions: {C} | Avg(10): {A} | Noise: {N}
```

**Quick Decode:**
```
Ep 5 | Reward: 325.4 | Outcome: GOAL | Goals: 2 | Collisions: 1 | Avg(10): 125.3 | Noise: 0.750

↓

Episode 5
├─ Total reward this episode: 325.4 (GOOD - goal reached)
├─ How it ended: GOAL (robot at target)
├─ Cumulative: 2 goals, 1 collision
├─ Recent 10 avg: 125.3 (trending up = GOOD)
└─ Exploration randomness: 0.750 (still exploring, not pure exploitation)
```

---

## Traffic Light System

### 🟢 GREEN (Everything Good)
```
Ep 10 | Reward: 412.1 | Outcome: GOAL | Goals: 5 | Collisions: 2 | Avg(10): 250.3 | Noise: 0.680
```
- ✅ Reward > 300: Goal reached
- ✅ Outcome: GOAL
- ✅ Avg(10) > 100: Strong learning
- ✅ Noise decreasing: Exploration working

---

### 🟡 YELLOW (Needs Attention)
```
Ep 10 | Reward: 45.3 | Outcome: GOAL | Goals: 1 | Collisions: 8 | Avg(10): 32.1 | Noise: 0.670
```
- ⚠️ Reward > 0 but low: weak progress reward
- ⚠️ Collisions growing: barrier may be too weak
- ⚠️ Avg(10) < 50: learning slower than expected
- 🔧 **Action:** Increase progress reward weight

---

### 🔴 RED (Fix Now)
```
Ep 10 | Reward: -245.3 | Outcome: COLLISION | Goals: 0 | Collisions: 10 | Avg(10): -120.3 | Noise: 0.670
```
- ❌ Reward < -200: collision heavy penalty
- ❌ Goals: 0: can't reach targets
- ❌ Avg(10) < 0: negative learning
- ❌ Collisions growing fast
- 🔧 **Action:** Check reward function + barrier, verify state space

---

## Real-Time Monitoring Checklist

### Every 5-10 Episodes
- [ ] Is `Avg(10)` increasing?
- [ ] Is `Outcome` mostly GOAL or TIMEOUT (not COLLISION)?
- [ ] Are `Goals` increasing?
- [ ] Is `Noise` decreasing steadily?

### Every Validation (every epoch/5000 steps)
- [ ] Is `Average Reward` > previous epoch?
- [ ] Is `Collision Rate` < previous epoch?
- [ ] Is there a "New best model found" message?

### Every 50 Episodes
- [ ] Check timestep/100 logs - is timing consistent?
- [ ] Is memory stable? (Time shouldn't increase dramatically)
- [ ] Total collision count not growing exponentially?

---

## One-Minute Log Analysis

```python
# Quick terminal command to monitor
while true; do
  tail -20 training.log | grep "Ep \|Validating\|Average Reward"
  sleep 5
done
```

**Key Lines to Watch:**
```
Ep 1 | Reward: -245.3 | Outcome: COLLISION | Goals: 0 | Avg(10): -245.3 | Noise: 0.800
                                                                ↑
                       Watch this trend upward over time

Validating at timestep 5000
Average Reward over 10 Evaluation Episodes, Epoch 1: 45.32
                                                      ↑
                       Watch this increase each epoch
```

---

## Outcome Meaning Matrix

| Outcome | Reward Range | Meaning | Action |
|---------|--------------|---------|--------|
| **GOAL** | +250 to +500 | Target reached | ✅ Good |
| **TIMEOUT** | -100 to +200 | Max steps reached | ⚠️ Slow navigation |
| **COLLISION** | -500 to -100 | Hit obstacle | ❌ Poor avoidance |

---

## Expected Training Curve

### Timesteps 0-5,000
```
Ep 1:  Avg(10): -245.3 | Noise: 0.800  (random exploration)
Ep 2:  Avg(10): 40.1   | Noise: 0.780  (finding direction)
Ep 3:  Avg(10): -25.4  | Noise: 0.760  (exploiting + exploring)
Ep 5:  Avg(10): 125.3  | Noise: 0.720  (good progress)
Validation Epoch 1: 45.32 reward  (baseline)
```

### Timesteps 5,000-50,000
```
Ep 20: Avg(10): 250.3  | Noise: 0.650  (consistent success)
Ep 50: Avg(10): 350.1  | Noise: 0.500  (strong policy)
Validation Epoch 5: 150.45 reward  (improving)
```

### Timesteps 50,000+
```
Ep 100: Avg(10): 420.5 | Noise: 0.250  (near convergence)
Ep 150: Avg(10): 460.3 | Noise: 0.050  (pure exploitation)
Validation Epoch 10: 287.32 reward  (converged)
```

---

## Emergency Debug Commands

### Extract Episode Metrics from Logs
```bash
# Get all episode lines
grep "Ep [0-9]" training.log | head -20

# Get validation results
grep "Average Reward" training.log

# Get goal achievements
grep "GOAL REACHED\|Outcome: GOAL" training.log | wc -l

# Get collision count
grep "Outcome: COLLISION" training.log | wc -l
```

### Plot Results
```python
import numpy as np
import matplotlib.pyplot as plt

evaluations = np.load("bots/bots/td3_rl/results/TD3_Mecanum.npy")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(evaluations, marker='o')
plt.xlabel("Validation Epoch")
plt.ylabel("Average Reward")
plt.title("Reward Progress")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(np.diff(evaluations), marker='s', color='orange')
plt.xlabel("Validation Epoch")
plt.ylabel("Reward Improvement")
plt.title("Epoch-to-Epoch Gain")
plt.grid()
plt.tight_layout()
plt.savefig("training_analysis.png")
```

---

## Metric Thresholds

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| Avg(10) Reward | > 200 | 50-200 | < 50 |
| Goal Success Rate | > 50% | 20-50% | < 20% |
| Collision Rate | < 0.2 | 0.2-0.4 | > 0.4 |
| Validation Reward Trend | Increasing | Flat | Decreasing |
| Noise Decay | Linear 0.8→0.05 | Stuck | Stuck at high |
| Time/100 Steps | 12-20s | 20-40s | > 40s |

---

## Common Issues & Fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Avg(10)` flat for 20+ eps | Learning stalled | ↑ learning_rate or ↑ progress_reward |
| `Goals: 0` after 10 eps | Can't navigate to goal | ↑ progress_reward weight |
| `Collisions` rapid growth | Hits walls often | ↑ barrier_reward weight or ↑ epsilon |
| `Noise` stuck at 0.800 | Exploration not decaying | ↓ expl_decay_steps in config |
| `Time/100` increasing fast | Memory leak | Check GPU memory, restart training |
| No "New best model" messages | No improvement | Review reward function scaling |
| `Collision Rate: 0.5` | Poor obstacle avoidance | ↑ barrier penalty cap from -2000 to -3000 |

---

## Save This For Reference

**Most Important Lines:**
1. Episode logs show `Avg(10)` - **this is your progress indicator**
2. Validation logs show `Average Reward` - **this is your true metric**
3. If `Avg(10)` is flat, training is stuck - **change reward weights**
4. If `Goals: 0`, robot can't reach targets - **increase progress reward**
5. If `Collisions` growing, obstacle avoidance broken - **increase barrier weight**

**You're doing well if:**
```
✅ Avg(10) > 100 by episode 20
✅ Validation Reward > 150 by epoch 3
✅ Goals reaching at least 30% of episodes
✅ Collisions < 20% of episodes
✅ Model saves increasing each epoch
```

