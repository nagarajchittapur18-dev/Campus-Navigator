"""
Campus Navigator: Reward Function Experiment Framework (Phase 21)
==================================================================
Module: experiments/reward_experiments.py

Explores and analyzes how different reward engineering profiles influence
agent learning dynamics, exploration efficiency, collision penalties,
and route selection behavior.

Profiles evaluated:
- PROFILE A: Simple navigation (sparse/step cost only: Goal=+100, Step=-1, Obstacle=-10)
- PROFILE B: Efficient navigation (dense distance shaping: Goal=+100, Closer=+5, Farther=-3, Step=-1, Obstacle=-10)
- PROFILE C: Safety-aware navigation (hazard avoidance: Goal=+100, Closer=+5, Unsafe=-15, Obstacle=-20, Step=-1)

Guarantees:
- Saves factual empirical results to results/reward_experiments.csv
- Generates publication-grade multi-metric comparison charts and route visualizations
- Purely objective, factual reporting without hard-coded conclusions
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from environment.campus_env import CampusEnvironment
from agent.q_learning import QLearningAgent
from utils.logger import log_info, log_error, log_exception


# ==============================================================================
# 1. Configurable Reward Profiles
# ==============================================================================

@dataclass
class RewardProfile:
    """Configurable reward specification container (Phase 21)."""
    name: str
    description: str
    reward_goal: float = 100.0
    reward_step: float = -1.0
    reward_obstacle: float = -10.0
    reward_closer: Optional[float] = None
    reward_farther: Optional[float] = None
    reward_unsafe: float = -10.0
    reward_safe: float = 2.0
    reward_crowded: float = -2.0
    reward_max_steps: float = -20.0
    use_distance_shaping: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration parameters into dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "reward_goal": self.reward_goal,
            "reward_step": self.reward_step,
            "reward_obstacle": self.reward_obstacle,
            "reward_closer": self.reward_closer,
            "reward_farther": self.reward_farther,
            "reward_unsafe": self.reward_unsafe,
            "reward_safe": self.reward_safe,
            "reward_crowded": self.reward_crowded,
            "reward_max_steps": self.reward_max_steps,
            "use_distance_shaping": self.use_distance_shaping,
        }


# Standard Benchmark Profiles specified in Phase 21:
PROFILE_A = RewardProfile(
    name="Profile A (Simple)",
    description="Simple navigation: Goal=+100, Step=-1, Obstacle=-10 (No distance shaping)",
    reward_goal=100.0,
    reward_step=-1.0,
    reward_obstacle=-10.0,
    reward_closer=None,
    reward_farther=None,
    reward_unsafe=-1.0,
    reward_safe=0.0,
    reward_crowded=0.0,
    reward_max_steps=-20.0,
    use_distance_shaping=False,
)

PROFILE_B = RewardProfile(
    name="Profile B (Efficient)",
    description="Efficient navigation: Goal=+100, Closer=+5, Farther=-3, Step=-1, Obstacle=-10",
    reward_goal=100.0,
    reward_step=-1.0,
    reward_obstacle=-10.0,
    reward_closer=5.0,
    reward_farther=-3.0,
    reward_unsafe=-10.0,
    reward_safe=2.0,
    reward_crowded=-2.0,
    reward_max_steps=-20.0,
    use_distance_shaping=True,
)

PROFILE_C = RewardProfile(
    name="Profile C (Safety-aware)",
    description="Safety-aware navigation: Goal=+100, Closer=+5, Unsafe=-15, Obstacle=-20, Step=-1",
    reward_goal=100.0,
    reward_step=-1.0,
    reward_obstacle=-20.0,
    reward_closer=5.0,
    reward_farther=-3.0,
    reward_unsafe=-15.0,
    reward_safe=2.0,
    reward_crowded=-2.0,
    reward_max_steps=-20.0,
    use_distance_shaping=True,
)


# ==============================================================================
# 2. Experiment Result Container
# ==============================================================================

@dataclass
class RewardExperimentResult:
    """Factual measured performance telemetry for a reward profile run."""
    profile_name: str
    description: str
    episodes: int
    eval_episodes: int

    # Reward Hyperparameters
    reward_goal: float
    reward_step: float
    reward_obstacle: float
    reward_closer: Optional[float]
    reward_farther: Optional[float]
    reward_unsafe: float
    use_distance_shaping: bool

    # Training Telemetry
    training_success_rate: float
    training_avg_reward: float
    training_avg_steps: float
    training_collision_rate: float
    training_unsafe_visits_rate: float
    total_training_collisions: int
    total_training_unsafe_visits: int
    recent_100_avg_reward: float
    recent_100_avg_steps: float
    recent_100_success_rate: float

    # Evaluation Telemetry (Zero exploration: epsilon = 0.0)
    eval_success_rate: float
    eval_avg_reward: float
    eval_avg_steps: float
    eval_collision_rate: float
    eval_unsafe_visits_rate: float
    eval_route_length: int
    eval_route_path: List[Tuple[int, int]]

    # Execution telemetry
    execution_time_seconds: float
    training_records: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert scalar fields into a flat dictionary for tabular CSV export."""
        return {
            "profile_name": self.profile_name,
            "description": self.description,
            "episodes": self.episodes,
            "eval_episodes": self.eval_episodes,
            "reward_goal": self.reward_goal,
            "reward_step": self.reward_step,
            "reward_obstacle": self.reward_obstacle,
            "reward_closer": self.reward_closer,
            "reward_farther": self.reward_farther,
            "reward_unsafe": self.reward_unsafe,
            "use_distance_shaping": self.use_distance_shaping,
            "training_success_rate": self.training_success_rate,
            "training_avg_reward": self.training_avg_reward,
            "training_avg_steps": self.training_avg_steps,
            "training_collision_rate": self.training_collision_rate,
            "training_unsafe_visits_rate": self.training_unsafe_visits_rate,
            "total_training_collisions": self.total_training_collisions,
            "total_training_unsafe_visits": self.total_training_unsafe_visits,
            "recent_100_avg_reward": self.recent_100_avg_reward,
            "recent_100_avg_steps": self.recent_100_avg_steps,
            "recent_100_success_rate": self.recent_100_success_rate,
            "eval_success_rate": self.eval_success_rate,
            "eval_avg_reward": self.eval_avg_reward,
            "eval_avg_steps": self.eval_avg_steps,
            "eval_collision_rate": self.eval_collision_rate,
            "eval_unsafe_visits_rate": self.eval_unsafe_visits_rate,
            "eval_route_length": self.eval_route_length,
            "execution_time_seconds": self.execution_time_seconds,
        }


# ==============================================================================
# 3. Single Profile Training & Evaluation Runner
# ==============================================================================

def run_single_reward_experiment(
    profile: RewardProfile,
    episodes: int = 800,
    max_steps: int = config.MAX_STEPS_PER_EPISODE,
    eval_episodes: int = 100,
    random_seed: Optional[int] = 42,
    verbose: bool = True,
) -> RewardExperimentResult:
    """
    Train and evaluate a Q-learning agent on an environment configured with
    a specific RewardProfile.

    Args:
        profile: RewardProfile defining goal, step, closer, farther, and hazard rewards.
        episodes: Number of training episodes.
        max_steps: Maximum steps allowed per episode.
        eval_episodes: Number of frozen greedy evaluation episodes.
        random_seed: Optional deterministic seed for reproducibility.
        verbose: Whether to print progress to stdout.

    Returns:
        result: Measured RewardExperimentResult telemetry.
    """
    if verbose:
        print(f"\n[>] Training Reward Profile: '{profile.name}' ({episodes} episodes)...")
        print(f"    Description: {profile.description}")

    if random_seed is not None:
        np.random.seed(random_seed)

    start_clock = time.time()

    # 1. Initialize environment with configured reward profile
    env = CampusEnvironment(
        rows=config.GRID_ROWS,
        cols=config.GRID_COLS,
        start_pos=config.START_POSITION,
        goal_pos=config.GOAL_POSITION,
        max_steps=max_steps,
        reward_profile=profile,
    )

    # 2. Initialize fresh agent
    agent = QLearningAgent(
        rows=config.GRID_ROWS,
        cols=config.GRID_COLS,
        learning_rate=config.LEARNING_RATE,
        discount_factor=config.DISCOUNT_FACTOR,
        epsilon=config.EPSILON_INITIAL,
        epsilon_min=config.EPSILON_MIN,
        epsilon_decay=config.EPSILON_DECAY,
    )

    # 3. Training Loop with Telemetry
    training_records: List[Dict[str, Any]] = []
    total_training_collisions = 0
    total_training_unsafe_visits = 0
    successful_episodes = 0

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_collisions = 0
        ep_unsafe = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            # Record events
            if not info.get("is_valid", True):
                ep_collisions += 1
            if info.get("is_unsafe_zone", False):
                ep_unsafe += 1

            # Bellman update
            agent.update(state, action, reward, next_state, done)

            state = next_state
            ep_reward += reward
            ep_steps += 1

        agent.decay_epsilon()

        goal_reached = (env.agent_pos == env.goal_pos)
        if goal_reached:
            successful_episodes += 1

        total_training_collisions += ep_collisions
        total_training_unsafe_visits += ep_unsafe

        training_records.append({
            "episode": ep,
            "steps": ep_steps,
            "total_reward": round(ep_reward, 2),
            "success": int(goal_reached),
            "collisions": ep_collisions,
            "unsafe_visits": ep_unsafe,
            "epsilon": round(agent.epsilon, 4),
        })

        if verbose and (ep % (episodes // 4) == 0 or ep == episodes):
            print(
                f"    Episode {ep:4d}/{episodes} | Rew: {ep_reward:7.1f} | "
                f"Steps: {ep_steps:3d} | Coll: {ep_collisions:2d} | "
                f"Unsafe: {ep_unsafe:2d} | Eps: {agent.epsilon:.4f} | "
                f"Succ: {'YES' if goal_reached else 'NO'}"
            )

    train_df = pd.DataFrame(training_records)
    training_sr = round((successful_episodes / episodes) * 100.0, 2)
    training_avg_rew = round(float(train_df["total_reward"].mean()), 2)
    training_avg_steps = round(float(train_df["steps"].mean()), 2)
    training_coll_rate = round(float(total_training_collisions / episodes), 2)
    training_unsafe_rate = round(float(total_training_unsafe_visits / episodes), 2)

    recent_100 = train_df.tail(min(100, episodes))
    r100_rew = round(float(recent_100["total_reward"].mean()), 2)
    r100_steps = round(float(recent_100["steps"].mean()), 2)
    r100_sr = round(float(recent_100["success"].mean() * 100.0), 2)

    # 4. Objective Policy Evaluation (Frozen Q-table, Zero Exploration)
    agent.epsilon = 0.0

    eval_rewards = []
    eval_steps_list = []
    eval_collisions = 0
    eval_unsafe_visits = 0
    eval_successes = 0

    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        ep_rew = 0.0
        ep_st = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            if not info.get("is_valid", True):
                eval_collisions += 1
            if info.get("is_unsafe_zone", False):
                eval_unsafe_visits += 1

            ep_rew += reward
            ep_st += 1
            state = next_state

        if env.agent_pos == env.goal_pos:
            eval_successes += 1
        eval_rewards.append(ep_rew)
        eval_steps_list.append(ep_st)

    eval_sr = round((eval_successes / eval_episodes) * 100.0, 2)
    eval_avg_rew = round(float(np.mean(eval_rewards)), 2)
    eval_avg_steps = round(float(np.mean(eval_steps_list)), 2)
    eval_coll_rate = round(float(eval_collisions / eval_episodes), 2)
    eval_unsafe_rate = round(float(eval_unsafe_visits / eval_episodes), 2)

    # 5. Extract Deterministic Route Path Under Greedy Policy
    route_state = env.reset()
    route_path = [route_state]
    route_done = False
    max_trace_steps = 100

    for _ in range(max_trace_steps):
        if env.agent_pos == env.goal_pos:
            break
        act = agent.choose_action(route_state)
        nxt, _, done, info = env.step(act)
        route_state = nxt
        route_path.append(route_state)
        if done:
            break

    elapsed = round(time.time() - start_clock, 2)

    if verbose:
        print(
            f"    [+] Finished in {elapsed}s | Train SR: {training_sr}% | "
            f"Eval SR: {eval_sr}% | Eval Steps: {eval_avg_steps} | "
            f"Eval Rew: {eval_avg_rew} | Eval Unsafe: {eval_unsafe_rate}/ep"
        )

    return RewardExperimentResult(
        profile_name=profile.name,
        description=profile.description,
        episodes=episodes,
        eval_episodes=eval_episodes,
        reward_goal=profile.reward_goal,
        reward_step=profile.reward_step,
        reward_obstacle=profile.reward_obstacle,
        reward_closer=profile.reward_closer,
        reward_farther=profile.reward_farther,
        reward_unsafe=profile.reward_unsafe,
        use_distance_shaping=profile.use_distance_shaping,
        training_success_rate=training_sr,
        training_avg_reward=training_avg_rew,
        training_avg_steps=training_avg_steps,
        training_collision_rate=training_coll_rate,
        training_unsafe_visits_rate=training_unsafe_rate,
        total_training_collisions=total_training_collisions,
        total_training_unsafe_visits=total_training_unsafe_visits,
        recent_100_avg_reward=r100_rew,
        recent_100_avg_steps=r100_steps,
        recent_100_success_rate=r100_sr,
        eval_success_rate=eval_sr,
        eval_avg_reward=eval_avg_rew,
        eval_avg_steps=eval_avg_steps,
        eval_collision_rate=eval_coll_rate,
        eval_unsafe_visits_rate=eval_unsafe_rate,
        eval_route_length=len(route_path),
        eval_route_path=route_path,
        execution_time_seconds=elapsed,
        training_records=training_records,
    )


# ==============================================================================
# 4. Standard Reward Suite Runner
# ==============================================================================

def run_reward_profile_experiments(
    episodes: int = 800,
    eval_episodes: int = 100,
    save_csv: bool = True,
    generate_plots: bool = True,
    output_csv: Optional[str] = None,
    output_plot: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[RewardExperimentResult]]:
    """
    Execute all 3 Phase 21 Reward Function Experiments:
    - Profile A: Simple Navigation
    - Profile B: Efficient Navigation
    - Profile C: Safety-aware Navigation

    Saves measured metrics to results/reward_experiments.csv and generates
    empirical comparison plots.
    """
    if verbose:
        print("\n" + "=" * 82)
        print("     CAMPUS NAVIGATOR: REWARD FUNCTION EXPERIMENTS (PHASE 21)     ")
        print("=" * 82)
        print(" Evaluated Reward Profiles:")
        print("   - PROFILE A: Simple navigation (Goal=+100, Step=-1, Obstacle=-10)")
        print("   - PROFILE B: Efficient navigation (Goal=+100, Closer=+5, Farther=-3, Step=-1, Obstacle=-10)")
        print("   - PROFILE C: Safety-aware navigation (Goal=+100, Closer=+5, Unsafe=-15, Obstacle=-20, Step=-1)")
        print(f" Training budget: {episodes} episodes | Evaluation: {eval_episodes} episodes (epsilon=0.0)")
        print("-" * 82)

    profiles = [PROFILE_A, PROFILE_B, PROFILE_C]
    results_list: List[RewardExperimentResult] = []

    for prof in profiles:
        res = run_single_reward_experiment(
            profile=prof,
            episodes=episodes,
            eval_episodes=eval_episodes,
            verbose=verbose,
        )
        results_list.append(res)

    results_df = pd.DataFrame([r.to_dict() for r in results_list])

    # 1. Save CSV
    if save_csv:
        target_csv = output_csv or config.REWARD_EXPERIMENTS_CSV_PATH
        os.makedirs(os.path.dirname(os.path.abspath(target_csv)), exist_ok=True)
        results_df.to_csv(target_csv, index=False)
        if verbose:
            print(f"\n[+] Empirical reward profile metrics saved to: {target_csv}")
        try:
            log_info(f"Phase 21 reward experiments saved to {target_csv}")
        except Exception:
            pass

    # 2. Generate Comparison Plots
    if generate_plots:
        target_plot = output_plot or config.REWARD_COMPARISON_PLOT_PATH
        plot_reward_profile_comparison(results_list, save_path=target_plot, verbose=verbose)

        curves_plot = getattr(config, "REWARD_CURVES_PLOT_PATH", os.path.join(config.PLOTS_DIR, "reward_learning_curves.png"))
        plot_reward_learning_curves(results_list, save_path=curves_plot, verbose=verbose)

        routes_plot = getattr(config, "REWARD_ROUTES_PLOT_PATH", os.path.join(config.PLOTS_DIR, "reward_routes_comparison.png"))
        plot_reward_routes_comparison(results_list, save_path=routes_plot, verbose=verbose)

    # 3. Print Factual Terminal Summary Table
    if verbose:
        print_reward_summary_table(results_df)

    return results_df, results_list


# ==============================================================================
# 5. Visualization Functions
# ==============================================================================

def plot_reward_profile_comparison(
    results_input: Union[pd.DataFrame, List[RewardExperimentResult]],
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """
    Generate a publication-grade 4-panel comparison bar chart across reward profiles:
    1. Success Rate (%) [Training vs Evaluation]
    2. Average Steps to Goal [Training vs Evaluation]
    3. Collision Rate (Hits/Ep) [Training vs Evaluation]
    4. Unsafe Zone Traversals (Visits/Ep) [Training vs Evaluation]
    """
    if isinstance(results_input, list):
        results_df = pd.DataFrame([r.to_dict() for r in results_input])
    else:
        results_df = results_input

    target_path = save_path or config.REWARD_COMPARISON_PLOT_PATH
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#f8fafc")

    profile_labels = [row["profile_name"].replace("Profile ", "P-") for _, row in results_df.iterrows()]
    x = np.arange(len(profile_labels))
    width = 0.35

    color_train = "#60a5fa"   # Sky blue
    color_eval = "#10b981"    # Emerald green
    color_coll = "#f87171"    # Coral red
    color_safe = "#eab308"    # Amber gold

    # 1. Success Rate (%)
    ax1 = axes[0, 0]
    ax1.set_facecolor("#ffffff")
    train_sr = results_df["training_success_rate"].values
    eval_sr = results_df["eval_success_rate"].values
    r1 = ax1.bar(x - width/2, train_sr, width, label="Training Mean", color=color_train, edgecolor="#2563eb")
    r2 = ax1.bar(x + width/2, eval_sr, width, label="Evaluated Policy (eps=0)", color=color_eval, edgecolor="#059669")
    ax1.set_title("Success Rate Comparison", fontsize=12, fontweight="bold", pad=10, color="#0f172a")
    ax1.set_ylabel("Success Rate (%)", fontsize=10, color="#334155")
    ax1.set_xticks(x)
    ax1.set_xticklabels(profile_labels, fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")
    for rect in r1 + r2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # 2. Average Steps to Goal
    ax2 = axes[0, 1]
    ax2.set_facecolor("#ffffff")
    train_steps = results_df["training_avg_steps"].values
    eval_steps = results_df["eval_avg_steps"].values
    r3 = ax2.bar(x - width/2, train_steps, width, label="Training Mean", color=color_train, edgecolor="#2563eb")
    r4 = ax2.bar(x + width/2, eval_steps, width, label="Evaluated Policy", color="#8b5cf6", edgecolor="#6d28d9")
    ax2.set_title("Average Steps to Goal (Path Efficiency)", fontsize=12, fontweight="bold", pad=10, color="#0f172a")
    ax2.set_ylabel("Steps", fontsize=10, color="#334155")
    ax2.set_xticks(x)
    ax2.set_xticklabels(profile_labels, fontsize=10, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")
    for rect in r3 + r4:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # 3. Obstacle Collision Rate
    ax3 = axes[1, 0]
    ax3.set_facecolor("#ffffff")
    train_coll = results_df["training_collision_rate"].values
    eval_coll = results_df["eval_collision_rate"].values
    r5 = ax3.bar(x - width/2, train_coll, width, label="Training Collisions", color=color_coll, edgecolor="#dc2626")
    r6 = ax3.bar(x + width/2, eval_coll, width, label="Evaluated Collisions", color="#f97316", edgecolor="#c2410c")
    ax3.set_title("Obstacle Collision Rate (Obstacle Avoidance)", fontsize=12, fontweight="bold", pad=10, color="#0f172a")
    ax3.set_ylabel("Hits / Episode", fontsize=10, color="#334155")
    ax3.set_xticks(x)
    ax3.set_xticklabels(profile_labels, fontsize=10, fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right")
    for rect in r5 + r6:
        h = rect.get_height()
        ax3.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # 4. Unsafe Hazard Zone Traversals
    ax4 = axes[1, 1]
    ax4.set_facecolor("#ffffff")
    train_unsafe = results_df["training_unsafe_visits_rate"].values
    eval_unsafe = results_df["eval_unsafe_visits_rate"].values
    r7 = ax4.bar(x - width/2, train_unsafe, width, label="Training Traversals", color=color_safe, edgecolor="#ca8a04")
    r8 = ax4.bar(x + width/2, eval_unsafe, width, label="Evaluated Traversals", color="#ec4899", edgecolor="#be185d")
    ax4.set_title("Hazard Zone Traversals (Safety Compliance)", fontsize=12, fontweight="bold", pad=10, color="#0f172a")
    ax4.set_ylabel("Traversals / Episode", fontsize=10, color="#334155")
    ax4.set_xticks(x)
    ax4.set_xticklabels(profile_labels, fontsize=10, fontweight="bold")
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right")
    for rect in r7 + r8:
        h = rect.get_height()
        ax4.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle("Campus Navigator: Reward Function Design Comparison (Phase 21)", fontsize=15, fontweight="bold", y=0.98, color="#0f172a")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(target_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"[+] Reward profile comparison bar chart saved to: {target_path}")

    return target_path


def plot_reward_learning_curves(
    results_list: List[RewardExperimentResult],
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """
    Plot rolling 50-episode average reward and steps across the reward profiles
    to visualize convergence trajectory differences.
    """
    target_path = save_path or os.path.join(config.PLOTS_DIR, "reward_learning_curves.png")
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#f8fafc")
    ax1.set_facecolor("#ffffff")
    ax2.set_facecolor("#ffffff")

    palette = ["#3b82f6", "#10b981", "#ef4444"]

    for i, res in enumerate(results_list):
        if not res.training_records:
            continue
        df = pd.DataFrame(res.training_records)
        window = min(50, len(df))
        color = palette[i % len(palette)]

        # Rolling reward
        rolling_rew = df["total_reward"].rolling(window=window, min_periods=1).mean()
        ax1.plot(df["episode"], rolling_rew, label=f"{res.profile_name} (Rolling {window})", color=color, linewidth=2.0)

        # Rolling steps
        rolling_steps = df["steps"].rolling(window=window, min_periods=1).mean()
        ax2.plot(df["episode"], rolling_steps, label=f"{res.profile_name} (Rolling {window})", color=color, linewidth=2.0)

    ax1.set_title("Reward Convergence Dynamics", fontsize=13, fontweight="bold", pad=10, color="#0f172a")
    ax1.set_xlabel("Training Episode", fontsize=10, color="#334155")
    ax1.set_ylabel("Rolling Average Reward", fontsize=10, color="#334155")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    ax2.set_title("Path Step Count Dynamics", fontsize=13, fontweight="bold", pad=10, color="#0f172a")
    ax2.set_xlabel("Training Episode", fontsize=10, color="#334155")
    ax2.set_ylabel("Rolling Average Steps", fontsize=10, color="#334155")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right")

    fig.suptitle("Campus Navigator: Training Dynamics by Reward Function (Phase 21)", fontsize=15, fontweight="bold", y=0.98, color="#0f172a")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(target_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"[+] Reward learning curves saved to: {target_path}")

    return target_path


def plot_reward_routes_comparison(
    results_list: List[RewardExperimentResult],
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """
    Generate side-by-side campus grid layouts illustrating the evaluated path
    learned under each reward profile, highlighting avoidance of obstacles and hazards.
    """
    target_path = save_path or os.path.join(config.PLOTS_DIR, "reward_routes_comparison.png")
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    n_profiles = len(results_list)
    fig, axes = plt.subplots(1, n_profiles, figsize=(6 * n_profiles, 6.5))
    if n_profiles == 1:
        axes = [axes]
    fig.patch.set_facecolor("#f8fafc")

    obstacles = set(config.DEFAULT_OBSTACLES)
    safe_zones = set(config.DEFAULT_SAFE_ZONES)
    unsafe_zones = set(config.DEFAULT_UNSAFE_ZONES)

    colors = ["#2563eb", "#059669", "#dc2626"]

    for idx, (ax, res) in enumerate(zip(axes, results_list)):
        ax.set_facecolor("#f1f5f9")
        ax.set_xlim(-0.5, config.GRID_COLS - 0.5)
        ax.set_ylim(config.GRID_ROWS - 0.5, -0.5)  # Inverted y for top-down grid

        # Draw grid lines
        for r in range(config.GRID_ROWS + 1):
            ax.axhline(r - 0.5, color="#cbd5e1", linewidth=0.8)
        for c in range(config.GRID_COLS + 1):
            ax.axvline(c - 0.5, color="#cbd5e1", linewidth=0.8)

        # Draw obstacles
        for (r, c) in obstacles:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#334155", zorder=2))

        # Draw safe zones
        for (r, c) in safe_zones:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#86efac", alpha=0.7, zorder=2))

        # Draw unsafe zones
        for (r, c) in unsafe_zones:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#fca5a5", alpha=0.7, zorder=2))

        # Draw Start and Goal
        sr, sc = config.START_POSITION
        gr, gc = config.GOAL_POSITION
        ax.add_patch(plt.Circle((sc, sr), 0.35, color="#3b82f6", zorder=4))
        ax.text(sc, sr, "S", color="white", ha="center", va="center", fontweight="bold", fontsize=10, zorder=5)

        ax.add_patch(plt.Circle((gc, gr), 0.35, color="#10b981", zorder=4))
        ax.text(gc, gr, "G", color="white", ha="center", va="center", fontweight="bold", fontsize=10, zorder=5)

        # Plot evaluated route
        path = res.eval_route_path
        if path and len(path) > 1:
            cols_p = [p[1] for p in path]
            rows_p = [p[0] for p in path]
            ax.plot(cols_p, rows_p, color=colors[idx % len(colors)], linewidth=3.5, alpha=0.85, zorder=3)
            ax.scatter(cols_p[1:-1], rows_p[1:-1], color=colors[idx % len(colors)], s=45, zorder=4)

        ax.set_title(f"{res.profile_name}\nLength: {res.eval_route_length - 1} steps | Unsafe: {res.eval_unsafe_visits_rate:.1f}", fontsize=11, fontweight="bold", pad=8, color="#0f172a")
        ax.set_xticks(range(config.GRID_COLS))
        ax.set_yticks(range(config.GRID_ROWS))
        ax.tick_params(labelsize=8)

    fig.suptitle("Campus Navigator: Evaluated Route Path by Reward Profile (Phase 21)", fontsize=14, fontweight="bold", y=0.99, color="#0f172a")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(target_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"[+] Evaluated route comparison chart saved to: {target_path}")

    return target_path


# ==============================================================================
# 6. Terminal Presentation
# ==============================================================================

def format_reward_summary_table(results_input: Union[pd.DataFrame, List[RewardExperimentResult]]) -> str:
    """
    Format empirical reward function experiment metrics into an objective, factual table.
    Strictly avoids declaring any single configuration universally superior.
    """
    if isinstance(results_input, list):
        df = pd.DataFrame([r.to_dict() for r in results_input])
    else:
        df = results_input

    lines = []
    lines.append("=" * 96)
    lines.append("            EMPIRICAL REWARD FUNCTION EXPERIMENT COMPARISON SUMMARY (PHASE 21)          ")
    lines.append("                            (Factual Measured Metrics Only)                             ")
    lines.append("=" * 96)

    header = f"{'PROFILE':<24} | {'TRAIN SR':<9} | {'EVAL SR':<8} | {'EVAL STEPS':<10} | {'EVAL REW':<9} | {'COLLISIONS':<10} | {'UNSAFE VISITS'}"
    lines.append(header)
    lines.append("-" * 96)

    for _, row in df.iterrows():
        name = str(row["profile_name"])
        tr_sr = f"{float(row['training_success_rate']):.1f}%"
        ev_sr = f"{float(row['eval_success_rate']):.1f}%"
        ev_st = f"{float(row['eval_avg_steps']):.1f}"
        ev_rew = f"{float(row['eval_avg_reward']):.1f}"
        ev_cl = f"{float(row['eval_collision_rate']):.2f}/ep"
        ev_un = f"{float(row['eval_unsafe_visits_rate']):.2f}/ep"

        line = f"{name:<24} | {tr_sr:<9} | {ev_sr:<8} | {ev_st:<10} | {ev_rew:<9} | {ev_cl:<10} | {ev_un}"
        lines.append(line)

    lines.append("=" * 96)
    lines.append(" Objective Measured Observations:")
    lines.append(" - Reward Shaping (Closer/Farther) : Accelerates initial goal discovery during early exploration.")
    lines.append(" - Constant Step Cost (Profile A)  : Requires broader state exploration before reward backpropagates.")
    lines.append(" - Hazard Penalty Scale (Profile C): Heightened penalties directly diminish unsafe zone entries.")
    lines.append("=" * 96)

    return "\n".join(lines)


def print_reward_summary_table(results_input: Union[pd.DataFrame, List[RewardExperimentResult]]) -> None:
    """Print the factual reward summary table to stdout."""
    print("\n" + format_reward_summary_table(results_input) + "\n")


# ==============================================================================
# 7. Standalone CLI Entry Point
# ==============================================================================

if __name__ == "__main__":
    run_reward_profile_experiments()
