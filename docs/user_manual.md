# User Manual & Operational Guide: Campus Navigator

Welcome to **Campus Navigator: An Intelligent Path-Finding Game Using Q-Learning**. This comprehensive user manual guides you through installation, game modes, controls, CLI options, and configuration settings.

---

## 1. System Requirements & Installation

### 1.1 Prerequisites
- **Python**: Version 3.8 or higher (Python 3.10+ recommended).
- **Operating System**: Windows 10/11, macOS, or Linux.
- **Display**: Minimum screen resolution of $1100 \times 750$ pixels (or headless driver for terminal training).

### 1.2 Step-by-Step Installation
1. Open a terminal in the project directory:
   ```bash
   cd "Campus Navigator"
   ```
2. (Recommended) Create and activate a clean virtual environment:
   ```bash
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
   *Dependencies: `pygame >= 2.5.0`, `numpy >= 1.24.0`, `matplotlib >= 3.7.0`, `pandas >= 2.0.0`.*

---

## 2. Launching the Interactive Game

To start the interactive graphical user interface:
```bash
python main.py
```

The game launches at $1100 \times 750$ resolution with a dark modern academic theme and an 8-option interactive main menu.

```text
============================================================
              CAMPUS NAVIGATOR: MAIN MENU                   
============================================================
 [1] Play as Human (WASD / Arrow Keys)
 [2] Watch Trained AI Navigate
 [3] Select Route / Landmark (Multi-Destination)
 [4] Training Results Dashboard & Plots
 [5] How Reinforcement Learning Works (Educational Viva)
 [6] Human vs. AI Performance Comparison
 [7] Configure Settings & Hyperparameters
 [8] Exit Campus Navigator
============================================================
```

---

## 3. Game Modes & Screen Navigation

### 3.1 Mode [1]: Play as Human
- **Objective**: Navigate the student agent from the Main Campus Gate `(0, 0)` to the Central Library `(9, 9)` while avoiding campus building blocks and hazardous construction zones.
- **Controls**:
  - `W` or `UP ARROW` : Move UP
  - `S` or `DOWN ARROW` : Move DOWN
  - `A` or `LEFT ARROW` : Move LEFT
  - `D` or `RIGHT ARROW` : Move RIGHT
  - `ESC` : Return to Main Menu
- **HUD Telemetry**:
  - **Steps Remaining**: Displays remaining step budget (out of 200).
  - **Cumulative Score**: Real-time reward tracking.
  - **Zone Alerts**: Visual notices when entering safe hubs (`+2`) or hazard zones (`-10`).
- **End State**: Upon arriving at the library, an automatic victory modal opens, leading to the **Human vs. AI Comparison** screen.

### 3.2 Mode [2]: Watch Trained AI Navigate
- **Objective**: Watch the pre-trained Q-Learning agent autonomously navigate the campus using its learned greedy policy ($\epsilon = 0.0$).
- **Controls**:
  - `SPACE` : Reset and replay navigation
  - `ESC` : Return to Main Menu
- **Visual Features**:
  - Step-by-step smooth path progression (throttled to human-observable speed).
  - Dynamic breadcrumb line connecting waypoints.
  - Glowing ripple arrival animation upon reaching the destination.

### 3.3 Mode [3]: Multi-Destination Landmark Selection (Phase 16)
- **Objective**: Choose among 5 predefined university landmarks:
  1. **Central Library** `(9, 9)`
  2. **Main Academic Block** `(1, 8)`
  3. **Computer Lab** `(4, 3)`
  4. **Student Cafeteria** `(3, 2)`
  5. **Sports Ground** `(9, 0)`
- **Interaction**: Select a destination button to dynamically load the goal-conditioned Q-tensor slice and watch the AI route to the chosen building.

### 3.4 Mode [4]: Training Results Dashboard (Phase 12)
- **Features**:
  - Displays key summary statistics: Training Episodes, Success Rate, Average Reward, Average Steps, Collision Rate, and Final Epsilon.
  - Displays high-resolution generated performance graphs:
    - Reward vs. Episode
    - Steps vs. Episode
    - Success Rate Progression
    - Collision Rate Trend
  - `[Open Results Folder]` button: Opens `results/` in the native system file explorer (Windows Explorer, macOS Finder, or Linux file manager).

### 3.5 Mode [5]: "How RL Works" Educational Viva Screen (Phase 13)
- **Features**:
  - Clear, concise definitions of all 10 fundamental RL concepts: Agent, Environment, State, Action, Reward, Q-Table, Exploration, Exploitation, Episode, and Learning.
  - Visual diagram of the closed-loop MDP cycle:
    $$\text{Environment} \to \text{State} \to \text{Agent} \to \text{Action} \to \text{Environment} \to \text{Reward} \to \text{Q-Update}$$
  - Full mathematical breakdown of the Bellman Q-Learning equation:
    $$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$
  - Tailored for university vivas, oral exams, and project demonstrations.

### 3.6 Mode [6]: Human vs. AI Performance Comparison (Phase 11)
- **Features**:
  - Side-by-side empirical comparison table contrasting:
    - Total Steps taken
    - Cumulative Reward scored
    - Path Length
    - Success Outcome
  - Neutral, factual presentation strictly avoiding biased claims of universal superiority.

### 3.7 Mode [7]: Settings Configuration (Phase 14)
- **Features**:
  - Interactive configuration screen reading and writing to `config.json`.
  - Configurable parameters:
    - Training Episodes
    - Learning Rate ($\alpha$)
    - Discount Factor ($\gamma$)
    - Initial Epsilon ($\epsilon_0$)
    - Minimum Epsilon ($\epsilon_{min}$)
    - Epsilon Decay Rate
    - Maximum Steps per Episode
    - AI Playback Speed
  - `[Save Settings]`, `[Reset Defaults]`, and `[Back]` buttons with full validation preventing negative episodes, discount $> 1$, or negative learning rates.

---

## 4. Command-Line Interface (CLI) Guide

Campus Navigator supports a comprehensive headless CLI for automated terminal workflows:

| CLI Command | Action Description |
|---|---|
| `python main.py` | Launches the interactive Pygame game window (default mode). |
| `python main.py --train` | Runs headless Q-learning training for 600 episodes and generates plots. |
| `python main.py --train --episodes 1000` | Runs headless training with a custom episode budget. |
| `python main.py --train --multi` | Trains the goal-conditioned Q-table across all 5 campus landmarks. |
| `python main.py --eval` | Evaluates the currently saved Q-table over 50 zero-exploration episodes. |
| `python main.py --eval --multi` | Evaluates the multi-destination policy across all 5 landmarks. |
| `python main.py --plot` | Re-generates all Matplotlib performance charts in `results/plots/`. |
| `python main.py --experiment` | Runs the Phase 20 benchmark suite (500, 1000, 2000 episodes) and comparison plots. |
| `python main.py --reward-exp` | Runs the Phase 21 reward profile experiments (Profiles A, B, C) and route overlays. |
| `python main.py --report` | Generates the official Phase 22 Final Evaluation Report (`final_report.json` & `final_results.csv`). |
| `python run_all_tests.py` | Executes the complete 75-test automated test suite across all 11 categories. |

---

## 5. Troubleshooting & Frequently Asked Questions

### 5.1 "No available video device" (Headless Environment)
- **Cause**: Pygame cannot find an active display monitor (common on remote SSH servers or Docker containers).
- **Solution**:
  - Use CLI commands which run completely headless without Pygame:
    ```bash
    python main.py --train
    python main.py --eval
    python main.py --experiment
    python main.py --report
    ```
  - For headless test execution, configure a dummy SDL driver:
    ```powershell
    # Windows PowerShell
    $env:SDL_VIDEODRIVER='dummy'
    python run_all_tests.py
    ```

### 5.2 "Missing Q-table" Alert in Game
- **Cause**: The pre-trained model file `models/q_table.npy` has not been generated or was deleted.
- **Solution**: Run headless training to generate the model:
  ```bash
  python main.py --train --episodes 1000
  ```

### 5.3 Resetting Corrupted Settings
- **Cause**: Manual edits to `config.json` introduced invalid syntax.
- **Solution**: Delete `config.json`. The application will automatically fall back to hardcoded defaults in `config.py` and recreate a clean configuration.

### 5.4 Diagnostic Logs
Detailed error stack traces and runtime information are automatically recorded in:
[`logs/campus_navigator.log`](file:///c:/Users/Nagaraj/Desktop/Campus%20Navigator/logs/campus_navigator.log)
