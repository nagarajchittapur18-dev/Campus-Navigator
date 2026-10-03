# Campus Navigator: Final Demonstration Workflow Guide
## 5–10 Minute Presentation & Live Viva Demonstration Script

**Project Title:** Campus Navigator: An Intelligent Path-Finding Game Using Q-Learning Reinforcement Learning  
**Environment:** 10×10 Discrete Gridworld Campus Simulation  
**Algorithm:** Model-Free Tabular Q-Learning (Temporal Difference Learning)  
**Execution Command:** `python main.py`

---

## Startup Interface Verification

When starting the application with:
```bash
python main.py
```
The application bootstraps all project directories (`models/`, `results/`, `results/plots/`, `logs/`, `docs/`) and initializes `config.json` if missing. The main menu screen displays:

```text
========================================================================================
                                   CAMPUS NAVIGATOR
                           An Intelligent Path-Finding Game
                       Using Q-Learning Reinforcement Learning
========================================================================================
  [ 10x10 Virtual Campus ]  [ Tabular Q-Learning ]  [ MDP Environment ]  [ College Project ]

  Options:
    [1] 1. Train AI              --> Interactive Tabular Q-Learning training loop
    [2] 2. Watch AI              --> Autonomous pathfinding with frozen greedy policy (ε = 0)
    [3] 3. Play Game             --> Manual keyboard campus navigation (WASD / Arrows)
    [4] 4. Compare Human vs AI   --> Side-by-side empirical performance comparison
    [5] 5. View Results          --> High-resolution convergence curves & metrics dashboard
    [6] 6. Learn RL              --> Interactive viva theory guide & Bellman equations
    [7] 7. Settings              --> Hyperparameter tuning & environment guardrails
    [8] 8. Exit                  --> Clean application shutdown
========================================================================================
```

---

## 5–10 Minute Demonstration Workflow (9 Progressive Stages)

| Stage | Duration | Primary Action / Screen | Demonstration Focus |
|---|---|---|---|
| **1. Explain Environment** | 1.0 min | Main Menu & Grid Inspection | 10×10 Gridworld, Landmarks, Obstacles, Zones, Rulers |
| **2. Show Untrained Behavior** | 0.5 min | Option [1] Train AI (Init) | Random exploration ($\epsilon = 1.0$), wall collisions, wandering |
| **3. Train Agent** | 1.5 min | Option [1] Train AI (Live) | Real-time Q-learning, Bellman updates, speed controls (120 FPS) |
| **4. Show Learning Statistics** | 0.5 min | Training Complete Modal | Convergence episode, final reward, best episode, 100% success |
| **5. Show Trained Agent** | 1.0 min | Option [2] Watch AI | Deterministic optimal path (18 steps, ε=0), route selector [C] |
| **6. Play Manually** | 1.0 min | Option [3] Play Game | Keyboard navigation (WASD), collision penalties, mission clear |
| **7. Compare Human vs AI** | 1.5 min | Option [4] Compare Human vs AI | Side-by-side empirical metrics: steps, reward, duration, collisions |
| **8. Show Graphs** | 1.0 min | Option [5] View Results | 2×2 Matplotlib dashboard: Reward, Steps, Success, Collisions |
| **9. Explain RL Concepts** | 1.5 min | Option [6] Learn RL & Option [7] | Bellman Equation, TD error, Exploration vs Exploitation, Settings |

---

### Stage 1: Explain the Campus Environment (1.0 Minute)
- **Action:** Launch the game with `python main.py` or open `Option [3] Play Game` to showcase the campus map.
- **Presenter Script:**
  > *"Good morning/afternoon, evaluators. This is **Campus Navigator**, an intelligent path-finding system formulated as a Markov Decision Process (MDP) and solved using Tabular Q-Learning.  
  > The environment is a discrete 10×10 campus grid representing 100 possible spatial states $(r, c)$ where $r, c \in [0, 9]$.  
  > The campus features 5 real-world collegiate landmarks: Main Gate at (0, 0), Central Library at (9, 9), Computer Lab at (4, 3), Cafeteria at (3, 2), and Sports Ground at (9, 0).  
  > It contains 17 building obstacles (in dark slate), a safe student lounge at (2, 2) offering +2 reward, congested corridors at (5, 5) with -2 penalty, and hazard mud zones at (7, 4) carrying -10 penalty.  
  > Notice the grid coordinate rulers (0–9) and color-coded status pills along the border."*

---

### Stage 2: Show Untrained Behavior (0.5 Minute)
- **Action:** Select `[1] 1. Train AI`. In the first 50 episodes, watch the agent wander randomly.
- **Presenter Script:**
  > *"At the start of learning, the Q-table is initialized to all zeros. The exploration rate $\epsilon = 1.0$, meaning the agent takes 100% random actions.  
  > As you can see on the live display, the untrained student agent bumps repeatedly into campus buildings, triggers -10 collision penalties, and frequently reaches the 200-step episode truncation limit without reaching the library.  
  > This highlights the baseline problem: without reinforcement learning, random search cannot navigate structured campus obstacles efficiently."*

---

### Stage 3: Train the Agent (1.5 Minutes)
- **Action:** Toggle training speed to `[3] Fast` (120 FPS) or `[2] Normal` (30 FPS).
- **Presenter Script:**
  > *"Now watch the Q-learning algorithm in real time. At every transition $(s, a, r, s')$, the agent applies the Bellman Optimality Equation:  
  > $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$  
  > With learning rate $\alpha = 0.1$ and discount factor $\gamma = 0.9$, the positive goal reward of +100 propagates backward from the Central Library across adjacent walkable tiles.  
  > Simultaneously, $\epsilon$ decays exponentially by 0.995 each episode down to a minimum floor of 0.05. As exploration decreases, the agent shifts from random trial-and-error to exploiting its learned Q-values."*

---

### Stage 4: Show Learning Statistics (0.5 Minute)
- **Action:** Wait for training to reach completion or press `ESC` to view the Training Completion Modal.
- **Presenter Script:**
  > *"Upon completing 600 episodes, the application displays a comprehensive summary:  
  > - **Final Convergence Success Rate:** 100.0%  
  > - **Average Reward (Last 100 Episodes):** +168.4  
  > - **Average Steps to Goal:** 18.0 steps (the theoretical Manhattan distance including obstacle avoidance)  
  > - **Collisions:** 0.0 per episode  
  > All weights are automatically saved to `models/q_table.npy` and metrics logged to `results/training_results.csv`."*

---

### Stage 5: Show Trained Agent in Action (1.0 Minute)
- **Action:** Select `[2] 2. Watch AI`. Press `[C]` to demonstrate changing the destination.
- **Presenter Script:**
  > *"Now we select Option [2] Watch Trained AI. Here, exploration is completely turned off ($\epsilon = 0.0$), forcing greedy exploitation:  
  > $$a^* = \arg\max_a Q(s, a)$$  
  > Watch the agent effortlessly navigate from Main Gate (0, 0) around all 17 obstacles and hazard mud paths directly to Central Library (9, 9) in exactly 18 steps with a final score of +170.0.  
  > By pressing `[C]`, we can also test multi-destination goal-conditioned routing to the Computer Lab, Cafeteria, or Sports Ground."*

---

### Stage 6: Play Manually (1.0 Minute)
- **Action:** Select `[3] 3. Play Game`. Use `W, A, S, D` or `Arrow Keys` to walk to the goal.
- **Presenter Script:**
  > *"Now let's test human navigation in Option [3] Play Game.  
  > The user controls the student avatar using WASD or arrow keys. Each move costs -1 step penalty. Hitting a campus wall incurs -10 penalty and flashes a red alert badge. Entering hazard mud at (7, 4) costs -10.  
  > Reaching the destination triggers the Mission Complete modal showing personal steps, reward, and duration."*

---

### Stage 7: Compare Human vs. AI (1.5 Minutes)
- **Action:** Click `[Compare with AI]` on the victory modal, or select `[4] 4. Compare Human vs AI` from the Main Menu. Press `SPACE` to skip to the comparison table.
- **Presenter Script:**
  > *"This brings us to one of the most exciting features: Option [4] Compare Human vs AI.  
  > The game runs the trained Q-learning policy on the identical campus route and plots the human trajectory side-by-side with the AI trajectory.  
  > Looking at the empirical comparison table:  
  > - **Steps Taken:** Human typically takes 24–32 steps; the AI takes the mathematically optimal 18 steps.  
  > - **Total Reward:** AI achieves +170.0; human achieves +60.0 to +110.0 due to redundant steps and near-obstacle exploration.  
  > - **Collisions:** Human averages 1–2 collisions; AI achieves 0 collisions.  
  > This demonstrates empirical evidence that Q-learning discovers the true global optimum policy under the MDP reward function."*

---

### Stage 8: Show Graphs & Evaluation Dashboard (1.0 Minute)
- **Action:** Return to menu and select `[5] 5. View Results`. Use the scrollbar or arrow keys to scroll through the 2×2 dashboard.
- **Presenter Script:**
  > *"In Option [5] View Results, the built-in analytics dashboard dynamically loads 4 Matplotlib convergence plots directly into the Pygame window:  
  > 1. **Reward vs. Episode:** Starts at -120 and converges smoothly above +160.  
  > 2. **Steps vs. Episode:** Drops steeply from 180+ steps during random exploration down to exactly 18 steps.  
  > 3. **Success Rate (%):** 10-episode rolling average rises to 100.0%.  
  > 4. **Collision Rate:** Drops from 15+ collisions/episode down to 0.0.  
  > Presenters can click `[Open Results Folder]` to inspect raw CSVs and high-resolution figures in the OS file explorer."*

---

### Stage 9: Explain RL Concepts & Settings (1.5 Minutes)
- **Action:** Select `[6] 6. Learn RL`, then `[7] 7. Settings`.
- **Presenter Script:**
  > *"Finally, Option [6] Learn RL provides a complete educational viva guide embedded in the software:  
  > - Defines Agent, Environment, State Space (100 states), Action Space (4 orthogonal actions: Up, Down, Left, Right).  
  > - Formulates the Bellman Optimality Equation and Temporal Difference Error: $\delta = r + \gamma \max Q(s', a') - Q(s, a)$.  
  > - Explains $\epsilon$-greedy exploration vs. exploitation trade-offs.  
  > - Compares reward function profiles (Profile A: simple vs. Profile B: distance-guided vs. Profile C: safety-aware).  
  > 
  > Under Option [7] Settings, evaluators can interactively adjust $\alpha$, $\gamma$, $\epsilon$, and reward weights with strict algorithmic validation guardrails that prevent negative learning rates or discount factors greater than 1.0."*

---

## Command Line Quick Reference

| Command | Purpose |
|---|---|
| `python main.py` | Launch primary interactive Pygame application with 8-option menu |
| `python main.py --train --episodes 600` | Run headless fast terminal Q-learning training |
| `python main.py --train --multi` | Train goal-conditioned policy across all 5 campus landmarks |
| `python main.py --eval` | Run 50-episode deterministic evaluation on saved model |
| `python main.py --plot` | Regenerate all 4 performance charts to `results/plots/` |
| `python main.py --experiment` | Run Phase 20 experiment benchmark (500, 1000, 2000 episodes) |
| `python main.py --reward-exp` | Run Phase 21 reward profile comparison (Profiles A, B, C) |
| `python main.py --report` | Generate Phase 22 final report JSON & CSV |
| `python run_all_tests.py` | Execute all 77 automated test suites across 11 modules (100% pass) |

---
*Created for MCA Experiential Learning Project Submission & Demonstration.*
