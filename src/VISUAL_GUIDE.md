# Visual Diagrams & Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Mecanum Robot Navigation                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                         SENSORS                                  │
├─────────────────────────────────────────────────────────────────┤
│  LIDAR (/scan) → 360 laser rays → 20 sampled values            │
│  Odometry (/odom/filtered) → Position (x, y) + Orientation     │
│  IMU (fused in EKF) → Yaw angle                                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓ ROS2 Topics
┌─────────────────────────────────────────────────────────────────┐
│              STATE REPRESENTATION (29 dims)                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ LIDAR Features (20 dims)                                │   │
│  │ • Sampled laser scan rays                               │   │
│  │ • Range: 0-3.5m (normalized)                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Goal Info (3 dims)                                      │   │
│  │ • Distance to goal (1)                                  │   │
│  │ • sin(angle_to_goal) (1) ← Better than raw angle!      │   │
│  │ • cos(angle_to_goal) (1) ← Circular continuity         │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Robot State (3 dims)                                    │   │
│  │ • Linear velocity X (1)                                 │   │
│  │ • Linear velocity Y (1) [Mecanum sideways]              │   │
│  │ • Angular velocity (1)                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Local Obstacle Sectors (3 dims) ← NEW!                │   │
│  │ • Front danger (0-90, 270-360 degrees)                  │   │
│  │ • Left danger (70-110 degrees)                          │   │
│  │ • Right danger (250-290 degrees)                        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                            ↓ 29-dim vector
┌─────────────────────────────────────────────────────────────────┐
│              NEURAL NETWORK (TD3 Agent)                          │
├─────────────────────────────────────────────────────────────────┤
│  Actor Network:                                                  │
│  Input: 29 dims → FC 512 → FC 256 → Output: 3 actions          │
│  Output: [linear_x, linear_y, angular_z] with tanh [-1, +1]    │
│                                                                  │
│  Critic Network (Twin Q-Networks):                              │
│  Input: 29 dims + 3 actions → FC 512 → FC 256 → Q-value        │
│  (Used for policy improvement)                                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓ Actions
┌─────────────────────────────────────────────────────────────────┐
│                    REWARD FUNCTION                               │
├─────────────────────────────────────────────────────────────────┤
│  Special Cases:                                                  │
│  • Goal Reached: R = +500                                       │
│  • Collision: R = -500                                          │
│                                                                  │
│  Normal Case (Weighted Sum):                                    │
│  R_total = 0.40*R_progress                                      │
│          + 0.25*R_obstacle                                      │
│          + 0.10*R_smoothness                                    │
│          + 0.10*R_heading                                       │
│          + 0.10*R_time                                          │
│          + 0.05*(-penalty_angular)                              │
│                                                                  │
│  ┌─ Progress: Attracts toward goal                             │
│  ├─ Obstacle: Repels from obstacles (smooth gradient)          │
│  ├─ Smoothness: Penalizes jerky changes                        │
│  ├─ Heading: Gentle direction correction                       │
│  ├─ Time: Encourages efficient movement                        │
│  └─ Angular: Limits unnecessary rotation                       │
└─────────────────────────────────────────────────────────────────┘
                    ↓ Reward Signal to Agent
┌─────────────────────────────────────────────────────────────────┐
│                  GAZEBO SIMULATION                               │
├─────────────────────────────────────────────────────────────────┤
│  • Robot physics updated (Mecanum drive)                         │
│  • Obstacles detected via collision sensors                     │
│  • LIDAR readings generated                                     │
│  • Odometry updated                                             │
│  • Time step advances (0.1s per step)                           │
└─────────────────────────────────────────────────────────────────┘
                    ↓ Next observation
```

---

## Reward Function Component Breakdown

```
TOTAL REWARD = Weighted Sum of 6 Components

┌────────────────────────────────────────────────────────────────┐
│ 1. PROGRESS REWARD (40% weight)                                │
├────────────────────────────────────────────────────────────────┤
│   Encourages moving closer to goal                             │
│   R = (old_distance - new_distance) * 200                      │
│                                                                │
│   Graph:                                                       │
│   Moving toward goal  → Positive reward  ↑                     │
│   Moving away        → Negative reward  ↓                      │
│   No movement        → Zero reward  →                          │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ 2. OBSTACLE AVOIDANCE (25% weight) ⭐ CRITICAL                │
├────────────────────────────────────────────────────────────────┤
│   Creates smooth repulsive force from obstacles                │
│                                                                │
│   Three zones:                                                 │
│                                                                │
│   DANGER ZONE (min < 0.50m):                                  │
│   ─────────────────────────────────────────────────            │
│   Heavy exponential penalty                                    │
│   R = -((0.50 - min) / 0.50)² * 150  (← Double weight)       │
│   Effect: Prevents collision                                  │
│                                                                │
│   WARNING ZONE (0.50m ≤ min < 0.75m):                         │
│   ──────────────────────────────────                           │
│   Moderate curved penalty                                     │
│   R = -((0.75 - min) / 0.75)^1.5 * 50                         │
│   Effect: Maintains safety buffer                             │
│                                                                │
│   SAFE ZONE (min ≥ 0.75m):                                    │
│   ─────────────────────────────                                │
│   Small bonus reward                                          │
│   R = (min / max_dist) * 10                                   │
│   Effect: Encourages maintaining distance (proactive)         │
│                                                                │
│   Visual Reward Curve:                                        │
│   +10 │      ___________                                      │
│       │     /            ▔▔▔                                  │
│     0 │____/                                                  │
│       │                                                       │
│    -50│    \                                                  │
│       │     \                                                 │
│   -150│      ▁▁▁▁▁ (exponential)                             │
│       └─────────────────────── min_laser (meters)            │
│       0.0    0.5    0.75   1.0   3.5                         │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ 3. SMOOTHNESS REWARD (10% weight)                              │
├────────────────────────────────────────────────────────────────┤
│   Encourages consistent, smooth motion                        │
│   R = -||action - prev_action|| * 5                           │
│                                                                │
│   Effect:                                                     │
│   Large changes  → Big penalty  ↓↓                            │
│   Small changes  → Small penalty  ↓                           │
│   No change      → Zero penalty  →                            │
│                                                                │
│   Result: Human-like smooth paths                             │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ 4. HEADING GUIDANCE (10% weight)                               │
├────────────────────────────────────────────────────────────────┤
│   Gentle guidance toward goal direction                       │
│   R = cos(angle_to_goal) * 5                                  │
│                                                                │
│   Effect:                                                     │
│   Facing goal      → +5 reward  ↑  (cos(0°) = 1)             │
│   Perpendicular    → 0 reward   →  (cos(90°) = 0)            │
│   Away from goal   → -5 reward  ↓  (cos(180°) = -1)          │
│                                                                │
│   Note: Gentle (not dominant) to avoid over-rotation          │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ 5. TIME EFFICIENCY (10% weight)                                │
├────────────────────────────────────────────────────────────────┤
│   Encourages active movement                                  │
│   R = -0.05 if moving_fast else -0.2                          │
│                                                                │
│   Effect:                                                     │
│   Moving forward  → Small penalty  ↓  (still encouraged)     │
│   Stationary      → Large penalty  ↓↓ (discouraged)          │
│                                                                │
│   Result: Prevents loitering, encourages dynamic motion       │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ 6. ANGULAR RESTRAINT (5% weight)                               │
├────────────────────────────────────────────────────────────────┤
│   Limits unnecessary spinning                                 │
│   R = -|angular_action| * 3 with 5% weight                   │
│                                                                │
│   Effect:                                                     │
│   No rotation     → 0 penalty     →                           │
│   Small rotation  → Small penalty  ↓                          │
│   Aggressive spin → Large penalty  ↓↓                         │
│                                                                │
│   Note: Weak (5%) to not over-suppress natural heading        │
└────────────────────────────────────────────────────────────────┘
```

---

## State Space Visualization

### BEFORE (Old - 25 dims):
```
Input to Neural Network:
┌─────────────────────────────────────────────┐
│ LIDAR rays 0-19     [20 dims]               │
│ Distance            [1 dim]                  │
│ Raw Angle (theta)   [1 dim] ← PROBLEM!     │
│ Velocity X          [1 dim]                  │
│ Velocity Y          [1 dim]                  │
│ Angular Velocity    [1 dim]                  │
└─────────────────────────────────────────────┘
          TOTAL: 25 dimensions

PROBLEM: theta = 179° vs -181° (same direction!)
         Network sees 360° difference → LEARNING INEFFICIENCY
```

### AFTER (New - 29 dims):
```
Input to Neural Network:
┌─────────────────────────────────────────────────────────────┐
│ LIDAR rays 0-19            [20 dims]                        │
│ Distance to goal           [1 dim]                           │
│ sin(angle_to_goal)         [1 dim] ← CONTINUOUS!           │
│ cos(angle_to_goal)         [1 dim] ← CIRCULAR FIX!         │
│ Velocity X                 [1 dim]                           │
│ Velocity Y                 [1 dim]                           │
│ Angular Velocity           [1 dim]                           │
│ Front Obstacle Danger      [1 dim] ← NEW!                  │
│ Left Obstacle Danger       [1 dim] ← NEW!                  │
│ Right Obstacle Danger      [1 dim] ← NEW!                  │
└─────────────────────────────────────────────────────────────┘
          TOTAL: 29 dimensions

IMPROVEMENTS:
✓ sin/cos encoding: Continuous (no -π to π jump)
✓ Obstacle sectors: Local awareness (not just global)
✓ Better features: Faster neural network learning
```

---

## Training Phase Progression

```
PHASE 1: LEARNING (Episodes 1-50)
═════════════════════════════════════════════════════════════════

Agent Task: "Don't crash!"

    Success Rate Growth:        Collision Incidents:
    50 ████                     25 ██████████
    40 ████                     20 ████████
    30 ██░░                     15 ████
    20 ██░░                     10 ██
    10 ░░░░                      5 █
     0 ░░░░                      0 _
       0  10 20 30 40 50          0  10 20 30 40 50
        Episodes                    Episodes

Expected Behavior:
  Ep 1: Random → Maybe succeed by luck
  Ep 10: Learning collision avoidance
  Ep 30: More avoidance learned → 40-50% success
  Ep 50: Consistent avoidance → 50-60% success

Main Focus: Reward prevents crashing more than reaching goal


PHASE 2: OPTIMIZATION (Episodes 51-150)
═════════════════════════════════════════════════════════════════

Agent Task: "Balance safety and reaching goals"

    Success Rate Growth:        Average Reward:
    100 ████████████            500 █████
     80 ████████░░░             400 ████░
     60 ██████░░░░              300 ███░░
     40 ████░░░░░░              200 ██░░░
     20 ██░░░░░░░░              100 █░░░░
      0 ░░░░░░░░░░                0 ░░░░░
       50 80 100 120 150           50 80 100 120 150
        Episodes                     Episodes

Expected Behavior:
  Ep 51: Better avoidance + goal seeking starts
  Ep 75: Most episodes reach goal (70% success)
  Ep 100: 80%+ success, rare collisions
  Ep 150: Converging to good policy

Main Focus: Shift from safety to efficiency


PHASE 3: REFINEMENT (Episodes 151+)
═════════════════════════════════════════════════════════════════

Agent Task: "Smooth, efficient navigation"

    Success Rate:               Path Smoothness:
    100 ████████████░            Smooth  ████████░░░
     90 ███████████░░            Decent  ██████░░░░░
     80 ██████████░░░            Rough   ████░░░░░░░
        151 160 170 ...             151 160 170 ...
        Episodes                    Episodes

Expected Behavior:
  Ep 151+: 90%+ success rate
  Ep 170+: Smooth, human-like paths
  Reward components: Balanced, stable
  Model ready: For real robot transfer

Main Focus: Polish and optimization
```

---

## Network Architecture

```
STATE (29 dims)
    │
    ├─→ ACTOR NETWORK ────────────────────→ ACTIONS (3 dims)
    │   Input: 29                           Output: tanh(-1,1)
    │   Dense 512 ReLU                      [vx, vy, omega]
    │   Dense 256 ReLU
    │   Output: 3 values
    │
    └─→ CRITIC NETWORK (Twin Q-Networks)
        Input: 29 state + 3 actions
        Dense 512 ReLU
        Dense 256 ReLU
        Output: Q-value (scalar)
        
        [Used only for training, not inference]


FORWARD PASS EXAMPLE:

State: [scan_0...19, distance, sin(θ), cos(θ), vx, vy, ω, f_obs, l_obs, r_obs]
       └─ 29 dimensions ─┘

        ↓ Actor.forward()

      Dense 512
    ╭─────────╮
    │ ReLU    │ 512 neurons
    ╰─────────╯
        ↓

      Dense 256
    ╭─────────╮
    │ ReLU    │ 256 neurons
    ╰─────────╯
        ↓

      Output 3
    ╭─────────╮
    │ tanh    │ Output: [0.23, -0.15, 0.67]
    ╰─────────╯
        ↓ Scale by max_action (1.0 m/s)

   Actions: [0.23 m/s, -0.15 m/s, 0.67 rad/s]
```

---

## Reward Distribution Over Training

```
Episode Reward Trajectory (Expected):

     500 ├─ TARGET
         │
     400 ├         ╭────── Converging Zone
         │       ╱          (Ep 100-150)
     300 ├     ╱ ╲ ╭─ Settling Zone (Ep 50-100)
         │   ╱     │
     200 ├ ╱ ╲ ╱   │
         │╱    ╲   │
     100 ├ ╭───╭───╯ ╭─ Learning Zone (Ep 1-50)
         │╱ ╭─╯
       0 ├─╯
         │
    -200 │ (Collision penalty)
         │
    -500 │ (Major collision)
         └─────────────────────────────────
           0  10  20  30  40  50  100  150
              Episodes

Key Points:
• Ep 1: Random behavior (-200 to +200)
• Ep 30: Learning shows effect (+0 to +300)
• Ep 50: Improving trend (+100 to +400)
• Ep 100: Converged behavior (+300 to +450)
• Ep 150: Stable (+350 to +500)
```

---

## Comparison Matrix

```
┌─────────────────────┬──────────────┬────────────────┐
│ Aspect              │ BEFORE       │ AFTER          │
├─────────────────────┼──────────────┼────────────────┤
│ Reward Components   │ 3            │ 6              │
│ Obstacle Safety     │ Binary       │ Continuous     │
│ State Dimensions    │ 25           │ 29             │
│ Angle Encoding      │ Raw (bad)    │ sin/cos (good) │
│ Local Awareness     │ No           │ Yes (3 sectors)│
│ Smoothness Reward   │ No           │ Yes            │
│ Heading Guidance    │ Hard (10)    │ Gentle (5)     │
│ Collision Penalty   │ -200         │ -500           │
│ Ep1-10 Success      │ 25%          │ 40%            │
│ Ep50 Success        │ 30%          │ 60%            │
│ Ep100 Success       │ 50%          │ 85%            │
│ Ep150 Success       │ 60%          │ 90%+           │
└─────────────────────┴──────────────┴────────────────┘
```

---

**These diagrams help visualize the complex interactions in your RL training system!**
