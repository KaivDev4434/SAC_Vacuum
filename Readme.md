### **Phase 1: Environment Setup**

**Objective:** Create a 2D environment with grid-based movement.

#### Tools:

- **Python + Gymnasium** (OpenAI Gym fork) for RL environment.
- **PyGame** or **Pyglet** for basic rendering.
- **NumPy** for grid state management.

#### Steps:

1. **Grid World:**

   - Create a `VacuumEnv` class inheriting from `gymnasium.Env`.
   - Define a grid (e.g., 10x10) where each cell can be:
     - `0`: Empty
     - `1`: Agent
     - `2`: Obstacle (static)
     - `3`: Dirt
   - Action space: Discrete (up, down, left, right, clean).
   - Observation space: Grid state (flattened or as a 2D array).

2. **Rendering:**

   - Visualize the grid with PyGame (colors for cells: white=empty, gray=obstacle, brown=dirt, red=agent).

3. **Basic Logic:**
   - Agent movement with boundary checks.
   - Collision detection with static obstacles.

---

### **Phase 2: Agent Coverage (Exploration)**

**Objective:** Train the agent to visit all tiles.

#### Reward Design:

- `+1` for visiting a new tile.
- `-0.1` for revisiting a tile.
- `+10` for full coverage.

#### Training:

- Start with **Q-Learning** or **DQN** (simpler baselines).
- Track visited tiles via a `coverage_map` matrix.

---

### **Phase 3: Static Obstacles & Dirt**

**Objective:** Add obstacles and dirt-spawning mechanics.

#### Implementation:

1. **Static Obstacles:**

   - Initialize obstacles randomly (e.g., 10% of grid cells).
   - Penalize collisions (`-1` reward).

2. **Dirt:**
   - Spawn dirt randomly (e.g., 20% of cells regenerate dirt every 50 steps).
   - Reward `+2` for cleaning dirt (action "clean" on a dirty cell).

---

### **Phase 4: Vision-Based Agent**

**Objective:** Restrict observations to a local FOV (field-of-view).

#### Observation Space:

- Replace full-grid observation with a local view (e.g., 5x5 window around the agent).
- Use **convolutional layers** in the RL model to process the FOV.

---

### **Phase 5: Dynamic Obstacles**

**Objective:** Add moving obstacles (e.g., pets, humans).

#### Implementation:

- Define dynamic obstacles with simple movement policies (e.g., random walk).
- Add collision detection and penalties (`-2` reward).
- Update rendering to show moving obstacles.

---

### **Phase 6: Continuous Motion**

**Objective:** Transition from grid-based to continuous motion.

#### Tools:

- **Box2D** (via `gymnasium`'s `Box2D` environments) for physics.
- **Action Space:** Continuous (e.g., `[dx, dy]` velocities in [-1, 1]).

#### Steps:

1. Redefine the environment with continuous coordinates.
2. Use raycasting for obstacle/dirt detection (simulate "vision").
3. Reward proximity to dirt (e.g., higher reward for closer cleaning).

---

### **Phase 7: Soft Actor-Critic (SAC) Implementation**

**Objective:** Train with SAC for continuous control.

#### Tools:

- **PyTorch** or **TensorFlow** for SAC implementation.
- **Stable-Baselines3** (SB3) for SAC baseline.

#### Steps:

1. **Custom Environment:**
   - Ensure the environment follows `gymnasium`’s API for continuous spaces.
2. **Hyperparameters:**
   - Learning rate: `3e-4`
   - Discount factor (`gamma`): `0.99`
   - Temperature parameter (`alpha`): Tuned automatically by SAC.
3. **Training:**
   - Use a replay buffer (size `1e6`).
   - Train for 500k steps, logging coverage and cleaning efficiency.

---

### **Phase 8: Benchmarking**

**Objective:** Compare SAC with other RL algorithms.

#### Algorithms to Test:

1. **DQN** (for discrete grid version).
2. **PPO** (continuous/discrete).
3. **A2C** (baseline).

#### Metrics:

- Coverage percentage over time.
- Dirt cleaned per episode.
- Collision rate with obstacles.

---

### **Timeline**

| Phase | Time Estimate |
| ----- | ------------- |
| 1–3   | 2 weeks       |
| 4–5   | 1 week        |
| 6     | 1.5 weeks     |
| 7–8   | 2 weeks       |

---

### **Pitfalls & Mitigation**

1. **Sparse Rewards:**
   - Use **intrinsic curiosity** or **reward shaping** (e.g., reward proximity to unvisited tiles).
2. **Continuous Control Complexity:**
   - Start with a small action space and simple physics.
3. **Vision Processing:**
   - Use CNNs with pretrained layers (e.g., ResNet) if FOV is large.

---

### **Final Deliverables**

1. A modular RL environment supporting grid/continuous modes.
2. Trained SAC model with metrics.
3. Comparative analysis of RL algorithms.
4. Visualization tools (e.g., trajectories, heatmaps).
