# Improvements Summary: Latest Changes

## [2026-01-07] Reward Function & Safety Overhaul

### 1. Shape-Aware Reciprocal Barrier Function (CBF)
**Problem Addressed:** 
The previous exponential barrier was too weak (~-1.2 per step) and assumed a circular robot, causing the bot to graze walls and get stuck in local minima.

**Solution Implemented:** 
A new **Shape-Aware** barrier in `gazebo_env.py` that respects the robot's rectangular hull (0.4m × 0.2m).

**Technical Details:**
- Calculates per-ray limits based on the robot's geometry (Limits(θ))
- **Front/Back Safety:** Activates at **0.45m** (braking distance)
- **Side Safety:** Activates at **0.25m** (corridor clearance)
- **Penalty:** Reciprocal function (1/d) capped at **-2000**
- **Result:** Strict wall avoidance with massive penalties (e.g., -1800) for violations, forcing the agent to stay in the safe zone

### 2. Lidar Indexing Fix
**Issue:** 
ROS `LaserScan` arrays typically start at -π (Back), not 0 (Front), causing misalignment in reward calculations.

**Fix Applied:** 
Corrected the angle calculation in `gazebo_env.py` to properly map the "back" of the robot to the appropriate array indices.

**Impact:** 
The reward function now correctly interprets laser readings relative to robot orientation, improving obstacle detection accuracy.

### 3. Configuration Updates
**File Modified:** `td3_config.yaml`

**Changes Made:**
- Synced parameters with loaded launch values (`expl_noise: 0.8`, `seed: 0`)
- Disabled `use_sim_time` for training stability

---

## Key Files Modified
- `bots/bots/td3_rl/gazebo_env.py` - Shape-aware barrier function + LIDAR indexing
- `td3_config.yaml` - Parameter synchronization

---

## 8. Quick Metrics Explanation

### What You'll See in Training Output:

```
[INFO] Ep 5 | Reward: 450 | Outcome: GOAL | Goals: 2 | Collisions: 0 | Avg(10): 280 | Noise: 0.95

Interpretation:
- Ep 5: Episode 5
- Reward: 450 = Good progress toward goal (450 < 500 because not direct path)
- Outcome: GOAL = Successfully reached target
- Goals: 2 = 2 goals reached so far
- Collisions: 0 = No crashes this episode
- Avg(10): 280 = Rolling average of last 10 episodes (good!)
- Noise: 0.95 = Exploration noise (decreases over time)

GOALS:
- Avg(10) > 300 = Consistently reaching goals
- Collisions < 1 per 10 episodes = Safe navigation
- Noise < 0.2 at Ep150 = Converged behavior
```

---

## 9. Future Enhancements

Once training stabilizes, consider:

1. **Dynamic Obstacles**
   - Add velocity to state
   - Implement collision prediction
   - Time-to-collision (TTC) metric

2. **Hierarchical Control**
   - Local planner (avoid obstacles)
   - Global planner (reach goal)
   - Combined policy network

3. **Curriculum Learning**
   - Start: Large arena, few obstacles
   - Mid: Smaller arena, more obstacles
   - Late: Challenging configurations

4. **Domain Randomization**
   - Variable obstacle sizes
   - Random world configurations
   - Different goal distances

---

## Success Criteria

✅ **Minimum**: 80% goal reach rate, <5% collision rate
✅ **Target**: 90% goal reach rate, <2% collision rate
✅ **Excellent**: 95% goal reach rate, smooth paths, <1% collision rate

**Estimated Training Time**:
- Phase 1 (Ep 1-50): 30-45 min
- Phase 2 (Ep 51-100): 30-45 min
- Phase 3 (Ep 101-150): 30-45 min
- **Total**: ~2-2.5 hours for CPU training

---

## Notes

- All weights are normalized (sum to 1.0)
- Collision penalty increased from -200 to -500 for clearer signal
- Progress reward reduced from 400 to 200 to allow obstacle awareness
- Exploration decay still active (expl_decay_steps: 500000)
- Model saves automatically when score improves

---

## [2026-01-07] Reward Function & Safety Overhaul

### 1. Shape-Aware Reciprocal Barrier Function (CBF)
- **Problem:** The previous exponential barrier was too weak ($~-1.2$ per step) and assumed a circular robot, causing the bot to graze walls and get stuck in local minima.
- **Solution:** Implemented a new **Shape-Aware** barrier in `gazebo_env.py` that respects the robot's rectangular hull ($0.4m \times 0.2m$).
- **Logic:**
  - Calculates per-ray limits based on the robot's geometry ($Limits(\theta)$).
  - **Front/Back Safety:** Activates at **0.45m** (braking distance).
  - **Side Safety:** Activates at **0.25m** (corridor clearance).
  - **Penalty:** Reciprocal function ($1/d$) capped at **-2000**.
- **Result:** Strict wall avoidance with massive penalties (e.g., $-1800$) for violations, forcing the agent to stay in the safe zone.

### 2. Lidar Indexing Fix
- **Fix:** Corrected the angle calculation in `gazebo_env.py`. ROS `LaserScan` arrays typically start at $-\pi$ (Back), not $0$ (Front).
- **Impact:** The reward function now correctly maps the "back" of the robot to the appropriate array indices.

### 3. Configuration Updates
- **File:** `td3_config.yaml`
- **Changes:**
  - Synced parameters with loaded launch values (`expl_noise: 0.8`, `seed: 0`).
  - Disabled `use_sim_time` for training stability.

---

## [2026-01-07] Reward Function & Safety Overhaul

### 1. Shape-Aware Reciprocal Barrier Function (CBF)
- **Problem:** The previous exponential barrier was too weak ($~-1.2$ per step) and assumed a circular robot, causing the bot to graze walls and get stuck in local minima.
- **Solution:** Implemented a new **Shape-Aware** barrier in `gazebo_env.py` that respects the robot's rectangular hull ($0.4m \times 0.2m$).
- **Logic:**
  - Calculates per-ray limits based on the robot's geometry ($Limits(\theta)$).
  - **Front/Back Safety:** Activates at **0.45m** (braking distance).
  - **Side Safety:** Activates at **0.25m** (corridor clearance).
  - **Penalty:** Reciprocal function ($1/d$) capped at **-2000**.
- **Result:** Strict wall avoidance with massive penalties (e.g., $-1800$) for violations, forcing the agent to stay in the safe zone.

### 2. Lidar Indexing Fix
- **Fix:** Corrected the angle calculation in `gazebo_env.py`. ROS `LaserScan` arrays typically start at $-\pi$ (Back), not $0$ (Front).
- **Impact:** The reward function now correctly maps the "back" of the robot to the appropriate array indices.

### 3. Configuration Updates
- **File:** `td3_config.yaml`
- **Changes:**
  - Synced parameters with loaded launch values (`expl_noise: 0.8`, `seed: 0`).
  - Disabled `use_sim_time` for training stability.
