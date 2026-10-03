# Reinforcement Learning Concepts & Viva Study Guide
===================================================

## 1. Project Overview & Context
**Campus Navigator** is an educational path-finding game built using **Tabular Q-Learning (Reinforcement Learning)**.
Instead of an abstract maze, the environment models a realistic **university campus gridworld**:
- **Campus Grid**: $10 \times 10$ discrete spatial layout (100 unique states).
- **Starting Landmark**: Campus Main Gate `(0, 0)`.
- **Target Destination**: Central Library `(9, 9)`.
- **Campus Obstacles**: 18 impassable university buildings, lecture complexes, and construction zones.
- **Specialized Zones**:
  - **Safe Student Hubs**: `(3, 2)` and `(7, 3)` (Cafeteria / Student Centre, small positive booster `+2`).
  - **Hazard Mud / Restricted Zones**: `(2, 5)`, `(5, 6)`, `(8, 7)` (Cautionary penalty `-10`).

---

## 2. Core RL Terminology Mapped to Code

| RL Concept | Theoretical Meaning | Implementation in Campus Navigator |
| :--- | :--- | :--- |
| **Agent** | The decision-maker learning from interaction. | `QLearningAgent` in `agent/q_learning.py` |
| **Environment** | The external world the agent navigates. | `CampusEnvironment` in `environment/campus_env.py` |
| **State ($S$)** | The current coordinate / situation of the agent. | Tuple `(row, col)` mapped to index $0 \dots 99$ via $s = \text{row} \times 10 + \text{col}$ |
| **Action ($A$)** | Cardinal decisions available to the agent. | 4 discrete moves: `0: UP`, `1: DOWN`, `2: LEFT`, `3: RIGHT` |
| **Reward ($R$)** | Feedback signal guiding the agent's behavior. | Goal: `+100`, Closer: `+5`, Farther: `-3`, Obstacle/Wall: `-10`, Hazard: `-10` |
| **Q-Value $Q(s, a)$**| Expected cumulative discounted reward of taking action $a$ in state $s$. | Stored in 2D NumPy array of shape `(100, 4)` |
| **Policy ($\pi$)** | Strategy mapping states to actions. | $\epsilon$-greedy during training; pure greedy $\arg\max_a Q(s, a)$ during evaluation |
| **Episode** | One full attempt from Main Gate until reaching Library or step limit. | Default maximum steps per episode = 200 |

---

## 3. The Q-Learning Mathematical Update Rule

Q-Learning is a **model-free, off-policy Temporal Difference (TD)** control algorithm.

The Bellman Optimality Equation used to update the Q-table on each step is:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

### Terms Explained:
- $\alpha$ (**Learning Rate**): Controls how much new experience overwrites old estimates ($0 < \alpha \le 1$). Configured to `0.1`.
- $\gamma$ (**Discount Factor**): Quantifies how much future rewards matter compared to immediate rewards ($0 \le \gamma < 1$). Configured to `0.9`.
- $R_{t+1}$: The immediate scalar reward received upon transitioning from $s$ to $s'$ via action $a$.
- $\max_{a'} Q(s', a')$: The maximum predicted reward possible from the next state $s'$. **Note: For terminal states (goal reached), this term is explicitly $0$.**
- $\left[ R_{t+1} + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$: **Temporal Difference (TD) Error**, measuring the surprise between expected reward and experienced reward.

---

## 4. Exploration vs. Exploitation ($\epsilon$-Greedy Strategy)

The fundamental dilemma in Reinforcement Learning:
- **Exploration**: Trying random or unknown actions to discover alternative pathways and avoid getting stuck in local optima.
- **Exploitation**: Choosing the action with the highest known Q-value $\arg\max_a Q(s, a)$ to maximize accumulated score.

### Exploration Schedule ($\epsilon$-Decay):
1. **Initial Rate**: $\epsilon = 1.0$ (100% random moves, discovering campus geometry).
2. **Decay Schedule**: $\epsilon \leftarrow \max(\epsilon_{min}, \epsilon \times \epsilon_{decay})$ after each episode.
   - $\epsilon_{decay} = 0.995$
   - $\epsilon_{min} = 0.05$ (guarantees 5% residual exploration during late training).
3. **Evaluation Mode**: $\epsilon = 0.0$ (Zero exploration, 100% greedy policy execution).

---

## 5. Potential Viva Questions & Model Answers

### Q1: Why is Q-Learning classified as "model-free"?
> **Answer**: Because the agent does not require prior mathematical knowledge of the environment's transition probability matrix $P(s' \mid s, a)$ or reward function $R(s, a)$. It learns optimal values purely through direct trial-and-error interaction.

### Q2: Why is Q-Learning classified as "off-policy"?
> **Answer**: Because the policy used to update the Q-value (greedy update choosing $\max_{a'} Q(s', a')$) is distinct from the behavior policy that selected the action (the exploratory $\epsilon$-greedy policy).

### Q3: What is the purpose of distance-based reward shaping?
> **Answer**: Sparse rewards (rewarding only when reaching the goal) can cause the agent to wander aimlessly for hundreds of episodes before encountering the library. Reward shaping gives immediate feedback based on Manhattan distance ($+5$ for moving closer, $-3$ for moving farther), providing a gradient that guides the agent toward the goal much faster.

### Q4: Why is future reward omitted at terminal states?
> **Answer**: At the terminal goal state, the episode ends; there are no subsequent actions or future transitions. Hence, $Q(s_{terminal}, a) = R_{goal}$, and adding $\gamma \max_{a'} Q(s', a')$ would incorrectly propagate non-existent future rewards.

### Q5: What is the theoretical shortest path in this campus layout?
> **Answer**: Due to the impassable buildings placed throughout the $10 \times 10$ campus grid, the Manhattan distance between `(0, 0)` and `(9, 9)` is 18 blocks. The optimal learned trajectory achieves exactly 18 steps with a cumulative reward of $+170.0$ and zero collisions.

### Q6: How does Human navigation compare empirically with the trained Q-Learning agent?
> **Answer**: In the Human vs AI comparison mode (Phase 11), both the human player and the trained agent navigate under identical rules, obstacles, and reward formulations. Human players demonstrate flexible real-time decision-making, but may take scenic routes or collide with obstacles while exploring the campus geometry. In contrast, after 1,000 training episodes, the tabular Q-Learning agent has converged to the Bellman-optimal policy, executing the theoretical minimum 18 steps (+170.0 cumulative return) with zero collisions. The comparison presents strictly empirical, measured results without making subjective claims of superiority.
