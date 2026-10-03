# Software Implementation & Engineering: Campus Navigator

This document provides a low-level engineering walkthrough of the source code, data structures, coordinate transformations, execution pipelines, and automated test harness of **Campus Navigator**.

---

## 1. Directory & Package Organization

The codebase is organized into modular Python packages with strict separation of concerns:

```text
Campus Navigator/
├── main.py                     # Primary entry point & CLI dispatcher
├── config.py                   # Central hyperparameters, campus layout & paths
├── config.json                 # User-customizable settings (Phase 14)
├── requirements.txt            # Minimal dependencies (pygame, numpy, matplotlib, pandas)
├── run_all_tests.py            # Master automated test runner (11 test suites, 75 tests)
│
├── environment/                # Phase 1 & 15: RL Markov Decision Process
│   ├── __init__.py
│   └── campus_env.py           # 10x10 Gridworld physics, obstacles, special zones, reward rules
│
├── agent/                      # Phase 2 & 16: Q-Learning Agent
│   ├── __init__.py
│   └── q_learning.py           # Tabular Q-Learning, epsilon-greedy decay, multi-goal tensor
│
├── training/                   # Phase 3, 4, & 22: Training & Evaluation Engines
│   ├── __init__.py
│   ├── train.py                # 1000-episode training loop, multi-destination trainer
│   ├── evaluate.py             # Frozen policy evaluation (epsilon=0.0)
│   └── final_evaluation.py     # Phase 22 final report generator (JSON & CSV)
│
├── experiments/                # Phase 20 & 21: Scientific Experimentation Framework
│   ├── __init__.py
│   ├── framework.py            # 500/1000/2000 episode horizon benchmarks & plotting
│   └── reward_experiments.py   # Profiles A, B, C reward design experiments & route visualizer
│
├── visualization/              # Phase 5: Publication Charts
│   ├── __init__.py
│   └── plots.py                # Reward, step, success rate, and collision Matplotlib plots
│
├── game/                       # Phase 6-17: Decoupled 2D Pygame Presentation Layer
│   ├── __init__.py
│   ├── game.py                 # Core game controller, 60 FPS clock, FSM state loop
│   ├── menu.py                 # 8 interactive screens, modals, dashboards, comparison tables
│   └── renderer.py             # 2D procedural rendering (coordinate rulers, HUD, animations)
│
├── utils/                      # Phase 18: Diagnostics & Logging
│   ├── __init__.py
│   └── logger.py               # Centralized structured UTF-8 logging to logs/
│
├── models/                     # Serialized Model Weights
│   ├── q_table.npy             # (100, 4) single-goal float32 array
│   └── q_table_multi.npy       # (5, 100, 4) goal-conditioned float32 array
│
├── results/                    # Generated Metrics & Charts
│   ├── training_results.csv    # 1000-episode historical training log
│   ├── evaluation_results.csv  # 100-episode frozen policy evaluation data
│   ├── evaluation_summary.json # High-level evaluation benchmarks
│   ├── experiments.csv         # Phase 20 horizon benchmark metrics
│   ├── reward_experiments.csv  # Phase 21 reward profile metrics
│   ├── final_report.json       # Phase 22 executive JSON evaluation report
│   ├── final_results.csv       # Phase 22 comprehensive CSV dataset
│   └── plots/                  # Generated PNG charts
│
├── logs/                       # Runtime Error & Audit Logs
│   └── campus_navigator.log    # Detailed execution logs with timestamps and stack traces
│
├── tests/                      # Phase 19: Comprehensive Automated Test Suite
│   ├── test_environment.py     # 9 tests: Gridworld physics, boundaries, obstacles, rewards
│   ├── test_agent.py           # 8 tests: Tabular memory, Bellman updates, epsilon decay
│   ├── test_training.py        # 3 tests: Training lifecycle, horizons, metrics logging
│   ├── test_evaluation.py      # 3 tests: Zero-epsilon policy, frozen weights guarantee
│   ├── test_game.py            # 5 tests: Menu navigation, human WASD, AI mode, alerts
│   ├── test_phase18_error_handling.py # 17 tests: All 10 operational failure scenarios
│   ├── test_multi_destinations.py     # 4 tests: 5 landmark destinations, tensor slicing
│   ├── test_phase17_ui.py      # 11 tests: Coordinate rulers, status pills, ripple animations
│   ├── test_experiments.py     # 5 tests: Phase 20 horizon benchmark execution & plots
│   ├── test_reward_experiments.py     # 6 tests: Phase 21 reward profile execution & plots
│   └── test_final_evaluation.py       # 4 tests: Phase 22 final report generation & schemas
│
└── docs/                       # Project Documentation
    ├── architecture.md
    ├── reinforcement_learning.md
    ├── implementation.md
    ├── experiments.md
    ├── user_manual.md
    └── rl_concepts.md
```

---

## 2. Core Class Specifications

### 2.1 `CampusEnvironment` (`environment/campus_env.py`)
```python
class CampusEnvironment:
    def __init__(
        self,
        rows: int = 10,
        cols: int = 10,
        start_pos: Tuple[int, int] = (0, 0),
        goal_pos: Tuple[int, int] = (9, 9),
        obstacles: Optional[List[Tuple[int, int]]] = None,
        max_steps: int = 200,
        reward_profile: Optional[RewardProfile] = None,
        ...
    ) -> None
```
- **Key Methods**:
  - `reset(start_pos=None) -> Tuple[int, int]`: Resets agent position, clears step count, returns starting state.
  - `step(action: int) -> Tuple[Tuple[int, int], float, bool, Dict]`: Transitions state based on action, returns $(s', r, \text{done}, \text{info})$.
  - `calculate_reward(current_state, next_state, is_valid, is_max_steps) -> float`: Evaluates immediate scalar feedback.
  - `is_valid_move(action, state=None) -> bool`: Checks grid boundary constraints and 19 building obstacles.
  - `set_destination(goal: Union[str, Tuple[int, int]]) -> Tuple[int, int]`: Reconfigures destination landmark coordinate (Phase 16).
  - `set_reward_profile(profile: RewardProfile) -> None`: Configures reward weights dynamically (Phase 21).

### 2.2 `QLearningAgent` (`agent/q_learning.py`)
```python
class QLearningAgent:
    def __init__(
        self,
        rows: int = 10,
        cols: int = 10,
        num_actions: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ) -> None
```
- **Data Attributes**:
  - `q_table`: `np.ndarray` of shape `(100, 4)` and dtype `float32`.
  - `multi_q_table`: `np.ndarray` of shape `(5, 100, 4)` and dtype `float32`.
- **Key Methods**:
  - `choose_action(state, valid_actions=None) -> int`: $\epsilon$-greedy action selection with uniform random tie-breaking.
  - `update(state, action, reward, next_state, done) -> float`: Bellman temporal difference update.
  - `decay_epsilon() -> float`: Multiplicative exploration decay with floor enforcement.
  - `save_model(filepath) -> None`: Serializes float32 array to `.npy` binary file.
  - `load_model(filepath) -> bool`: Deserializes weights with corruption, shape, and NaN/Inf validation.

---

## 3. Coordinate Systems & UI Mapping

The game window operates at an optimal academic resolution of **$1100 \times 750$ pixels**:

```text
(0,0) ┌───────────────────────────────────────────────────────────── (1100, 0)
      │  HEADER / STATUS BAR (Title, Mode, Route, Active Profiles)
      ├───────────────────────────────┬─────────────────────────────┤
      │                               │  SIDEBAR DASHBOARD (HUD)     │
      │   CAMPUS 10x10 GRID           │  - Step Budget & Gauges     │
      │   (Cell size: 56x56 px)       │  - Cumulative Reward        │
      │   Grid Canvas: 560x560 px     │  - Exploration (Epsilon)    │
      │   Offset: (X=40, Y=110)       │  - Campus Zone Legend       │
      │   Coordinate Rulers (0..9)    │  - Controls & Action Log    │
      │                               │                             │
      └───────────────────────────────┴─────────────────────────────┘ (1100, 750)
```

### Coordinate Transformation Functions (`game/renderer.py`)
- Grid state $(r, c)$ to Screen Pixel $(x, y)$:
  $$x = \text{GRID\_OFFSET\_X} + c \times \text{CELL\_SIZE}$$
  $$y = \text{GRID\_OFFSET\_Y} + r \times \text{CELL\_SIZE}$$
  where $\text{GRID\_OFFSET\_X} = 40$, $\text{GRID\_OFFSET\_Y} = 110$, and $\text{CELL\_SIZE} = 56\text{ px}$.
- Screen Pixel $(x, y)$ to Grid state $(r, c)$:
  $$c = \lfloor (x - \text{GRID\_OFFSET\_X}) / \text{CELL\_SIZE} \rfloor$$
  $$r = \lfloor (y - \text{GRID\_OFFSET\_Y}) / \text{CELL\_SIZE} \rfloor$$

---

## 4. Automated Test Harness (`tests/`)

Campus Navigator contains **75 automated unit and regression tests** organized into 11 test modules:

| Test Module | Coverage Scope | Test Count |
|---|---|---|
| [`tests/test_environment.py`](file:///c:/Users/Nagaraj/Desktop/Campus%20Navigator/tests/test_environment.py) | MDP grid physics, wall blocking, obstacle collision, goal detection, special zone rewards | 9 |
| [`tests/test_agent.py`](file:///c:/Users/Nagaraj/Desktop/Campus%20Navigator/tests/test_agent.py) | Q-table initialization, Bellman update rule, epsilon decay schedule, weight persistence | 8 |
| [`tests/test_training.py`](file:///c:/Users/Nagaraj/Desktop/Campus%20Navigator/tests/test_training.py) | Training loop start/finish, episode horizons, metrics logging to CSV | 3 |
| [`tests/test_evaluation.py`](file:///c:/Users/Nagaraj/Desktop/Campus%20Navigator/tests/test_evaluation.py) | Pretrained model loading, frozen Q-table verification (zero mutation), metrics calculation | 3 |
| [`tests/test_game.py`](file:///c:/Users/Nagaraj/Desktop/Campus%20Navigator/tests/test_game.py) | Window init, menu transitions, human WASD movement, Watch AI playback, missing model handling | 5 |
| [`tests/test_phase18_error_handling.py`](file:///c:/Users/Nagaraj/Desktop/Campus%20Navigator/tests/test_phase18_error_handling.py) | All 10 major failure scenarios: missing files, corrupted tensors, invalid actions, headless crashes | 17 |
| [`tests/test_multi_destinations.py`](file:///c:/Users/Nagaraj/Desktop/Campus%20Navigator/tests/test_multi_destinations.py) | 5 campus destinations, goal-conditioned tensor `(5, 100, 4)`, destination modal | 4 |
| [`tests/test_phase17_ui.py`](file:///c:/Users/Nagaraj/Desktop/Campus%20Navigator/tests/test_phase17_ui.py) | Coordinate rulers, pulsing status badges, training progress bars, numbered waypoint markers | 11 |
| [`tests/test_experiments.py`](file:///c:/Users/Nagaraj/Desktop/Campus%20Navigator/tests/test_experiments.py) | Phase 20 horizon benchmarks (500/1000/2000 eps), metrics calculations, CSV/plot persistence | 5 |
| [`tests/test_reward_experiments.py`](file:///c:/Users/Nagaraj/Desktop/Campus%20Navigator/tests/test_reward_experiments.py) | Phase 21 reward profiles (A, B, C), reward calculations, hazard tracking, comparison plots | 6 |
| [`tests/test_final_evaluation.py`](file:///c:/Users/Nagaraj/Desktop/Campus%20Navigator/tests/test_final_evaluation.py) | Phase 22 final report generation, JSON schema validation, CSV dataset generation | 4 |

### Running the Entire Suite Headless:
```bash
python run_all_tests.py
```
*Guaranteed 100% pass rate (75/75 passing, 0 failures, 0 errors).*
