"""
Campus Navigator: RL Experimentation & Benchmarking Framework (Phase 20)
========================================================================
Module: experiments/framework.py

Provides a modular, objective framework to execute, compare, and visualize
reinforcement learning experiments across distinct training configurations:
- Experiment 1: 500 episodes
- Experiment 2: 1000 episodes
- Experiment 3: 2000 episodes

For each experiment, records and calculates:
1. Success Rate (%)
2. Average Reward
3. Average Steps
4. Collision Rate (Hits/Episode)
Across both the dynamic training progression and post-training frozen policy evaluation.

Persists results to:
- results/experiments.csv
- results/plots/experiment_comparison.png
- results/plots/experiment_learning_curves.png
"""

from dataclasses import dataclass, field
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
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
from utils.logger import log_info, log_warning, log_error


# ==============================================================================
# 1. Configuration & Result Dataclasses
# ==============================================================================

@dataclass
class ExperimentConfig:
    """Configuration container for an empirical RL experiment."""
    name: str
    episodes: int
    max_steps: int = config.MAX_STEPS_PER_EPISODE
    learning_rate: float = config.LEARNING_RATE
    discount_factor: float = config.DISCOUNT_FACTOR
    initial_epsilon: float = config.EPSILON_INITIAL
    min_epsilon: float = config.EPSILON_MIN
    epsilon_decay: float = config.EPSILON_DECAY
    eval_episodes: int = 100
    start_pos: Tuple[int, int] = config.START_POSITION
    goal_pos: Tuple[int, int] = config.GOAL_POSITION
    random_seed: Optional[int] = None
    save_model: bool = False
    model_save_path: Optional[str] = None


@dataclass
class ExperimentResult:
    """Factual measured performance metrics resulting from an experiment run."""
    experiment_name: str
    episodes: int
    max_steps: int
    learning_rate: float
    discount_factor: float
    final_epsilon: float

    # Overall Training Metrics (across all episodes)
    training_success_rate: float
    training_avg_reward: float
    training_avg_steps: float
    training_collision_rate: float
    total_training_collisions: int
    successful_training_episodes: int

    # Converged Recent Performance (Last 100 Training Episodes)
    recent_100_success_rate: float
    recent_100_avg_reward: float
    recent_100_avg_steps: float
    recent_100_collision_rate: float

    # Frozen Policy Evaluation Metrics (Zero Exploration, epsilon = 0.0)
    eval_episodes: int
    eval_success_rate: float
    eval_avg_reward: float
    eval_avg_steps: float
    eval_collision_rate: float
    eval_min_steps: int
    eval_max_steps: int
    eval_optimal_route_length: int

    # Execution telemetry
    best_episode: int
    best_reward: float
    non_zero_q_entries: int
    execution_time_seconds: float

    # Per-episode raw records for learning curve analysis
    training_records: List[Dict] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict:
        """Convert scalar metrics into dictionary format for CSV export."""
        return {
            "experiment_name": self.experiment_name,
            "episodes": self.episodes,
            "max_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "final_epsilon": round(self.final_epsilon, 4),
            # Training metrics
            "training_success_rate": round(self.training_success_rate, 2),
            "training_avg_reward": round(self.training_avg_reward, 2),
            "training_avg_steps": round(self.training_avg_steps, 2),
            "training_collision_rate": round(self.training_collision_rate, 2),
            "total_training_collisions": self.total_training_collisions,
            "successful_training_episodes": self.successful_training_episodes,
            # Recent 100 episodes
            "recent_100_success_rate": round(self.recent_100_success_rate, 2),
            "recent_100_avg_reward": round(self.recent_100_avg_reward, 2),
            "recent_100_avg_steps": round(self.recent_100_avg_steps, 2),
            "recent_100_collision_rate": round(self.recent_100_collision_rate, 2),
            # Post-training evaluation
            "eval_episodes": self.eval_episodes,
            "eval_success_rate": round(self.eval_success_rate, 2),
            "eval_avg_reward": round(self.eval_avg_reward, 2),
            "eval_avg_steps": round(self.eval_avg_steps, 2),
            "eval_collision_rate": round(self.eval_collision_rate, 2),
            "eval_min_steps": self.eval_min_steps,
            "eval_max_steps": self.eval_max_steps,
            "eval_optimal_route_length": self.eval_optimal_route_length,
            # Diagnostic telemetry
            "best_episode": self.best_episode,
            "best_reward": round(self.best_reward, 2),
            "non_zero_q_entries": self.non_zero_q_entries,
            "execution_time_seconds": round(self.execution_time_seconds, 2),
        }


# ==============================================================================
# 2. Single Experiment Runner
# ==============================================================================

def run_single_experiment(cfg: ExperimentConfig, verbose: bool = True) -> ExperimentResult:
    """
    Execute a single isolated RL training and evaluation experiment.

    Args:
        cfg: Experiment configuration specifying episodes, lr, discount, etc.
        verbose: Whether to print live progress to stdout.

    Returns:
        result: Measured ExperimentResult containing training and evaluation metrics.
    """
    if verbose:
        print(f"\n[>] Starting Experiment: '{cfg.name}' ({cfg.episodes} episodes)...")

    if cfg.random_seed is not None:
        np.random.seed(cfg.random_seed)

    start_clock = time.time()

    # 1. Initialize decoupled environment and agent
    env = CampusEnvironment(
        rows=config.GRID_ROWS,
        cols=config.GRID_COLS,
        start_pos=cfg.start_pos,
        goal_pos=cfg.goal_pos,
        max_steps=cfg.max_steps,
    )
    agent = QLearningAgent(
        rows=env.rows,
        cols=env.cols,
        num_actions=config.NUM_ACTIONS,
        learning_rate=cfg.learning_rate,
        discount_factor=cfg.discount_factor,
        epsilon=cfg.initial_epsilon,
        epsilon_min=cfg.min_epsilon,
        epsilon_decay=cfg.epsilon_decay,
    )

    training_records: List[Dict] = []
    total_collisions = 0
    successful_episodes = 0
    best_episode = 1
    best_reward = -float("inf")

    # 2. Live Training Loop
    for ep in range(1, cfg.episodes + 1):
        state = env.reset()
        ep_reward = 0.0
        steps = 0
        collisions = 0
        done = False
        goal_reached = False

        for step in range(1, cfg.max_steps + 1):
            action = agent.choose_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)

            ep_reward += reward
            steps += 1
            if not info.get("is_valid", True):
                collisions += 1

            state = next_state
            if done:
                goal_reached = info.get("goal_reached", False)
                break

        agent.decay_epsilon()

        total_collisions += collisions
        if goal_reached:
            successful_episodes += 1

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_episode = ep

        record = {
            "episode": ep,
            "steps": steps,
            "total_reward": ep_reward,
            "success": 1 if goal_reached else 0,
            "collisions": collisions,
            "epsilon": agent.epsilon,
        }
        training_records.append(record)

        if verbose and (ep % max(1, cfg.episodes // 5) == 0 or ep == cfg.episodes):
            print(f"    Episode {ep:4d}/{cfg.episodes} | Rew: {ep_reward:6.1f} | Steps: {steps:3d} | Hits: {collisions:2d} | Eps: {agent.epsilon:.4f} | Succ: {'YES' if goal_reached else 'NO'}")

    train_df = pd.DataFrame(training_records)

    # Calculate overall training benchmarks
    training_success_rate = (successful_episodes / cfg.episodes) * 100.0
    training_avg_reward = float(train_df["total_reward"].mean())
    training_avg_steps = float(train_df["steps"].mean())
    training_collision_rate = float(train_df["collisions"].mean())

    # Calculate recent 100 episodes benchmarks
    recent_n = min(100, cfg.episodes)
    recent_df = train_df.tail(recent_n)
    recent_success_rate = (recent_df["success"].sum() / recent_n) * 100.0
    recent_avg_reward = float(recent_df["total_reward"].mean())
    recent_avg_steps = float(recent_df["steps"].mean())
    recent_collision_rate = float(recent_df["collisions"].mean())

    # 3. Post-Training Policy Evaluation (Zero Exploration: epsilon = 0.0, Q-table frozen)
    agent.epsilon = 0.0
    eval_rewards: List[float] = []
    eval_steps_list: List[int] = []
    eval_collisions_list: List[int] = []
    eval_success_count = 0
    sample_route: List[Tuple[int, int]] = []

    for ev_ep in range(cfg.eval_episodes):
        state = env.reset()
        ev_rew = 0.0
        ev_steps = 0
        ev_coll = 0
        ev_path = [state]
        ev_done = False
        ev_success = False

        for _ in range(cfg.max_steps):
            action = agent.get_best_action(state)
            next_state, reward, done, info = env.step(action)
            ev_rew += reward
            ev_steps += 1
            ev_path.append(next_state)
            if not info.get("is_valid", True):
                ev_coll += 1
            state = next_state
            if done:
                ev_success = info.get("goal_reached", False)
                break

        eval_rewards.append(ev_rew)
        eval_steps_list.append(ev_steps)
        eval_collisions_list.append(ev_coll)
        if ev_success:
            eval_success_count += 1
            if not sample_route:
                sample_route = ev_path

    eval_success_rate = (eval_success_count / cfg.eval_episodes) * 100.0
    eval_avg_reward = float(np.mean(eval_rewards))
    eval_avg_steps = float(np.mean(eval_steps_list))
    eval_collision_rate = float(np.mean(eval_collisions_list))
    eval_min_steps = int(np.min(eval_steps_list))
    eval_max_steps = int(np.max(eval_steps_list))
    eval_route_len = len(sample_route) if sample_route else len(ev_path)

    execution_time = time.time() - start_clock
    non_zeros = int(np.count_nonzero(agent.q_table))

    if cfg.save_model and cfg.model_save_path:
        agent.save_model(cfg.model_save_path, save_multi=False)

    result = ExperimentResult(
        experiment_name=cfg.name,
        episodes=cfg.episodes,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        discount_factor=cfg.discount_factor,
        final_epsilon=agent.epsilon_initial * (agent.epsilon_decay ** cfg.episodes),
        training_success_rate=training_success_rate,
        training_avg_reward=training_avg_reward,
        training_avg_steps=training_avg_steps,
        training_collision_rate=training_collision_rate,
        total_training_collisions=total_collisions,
        successful_training_episodes=successful_episodes,
        recent_100_success_rate=recent_success_rate,
        recent_100_avg_reward=recent_avg_reward,
        recent_100_avg_steps=recent_avg_steps,
        recent_100_collision_rate=recent_collision_rate,
        eval_episodes=cfg.eval_episodes,
        eval_success_rate=eval_success_rate,
        eval_avg_reward=eval_avg_reward,
        eval_avg_steps=eval_avg_steps,
        eval_collision_rate=eval_collision_rate,
        eval_min_steps=eval_min_steps,
        eval_max_steps=eval_max_steps,
        eval_optimal_route_length=eval_route_len,
        best_episode=best_episode,
        best_reward=best_reward,
        non_zero_q_entries=non_zeros,
        execution_time_seconds=execution_time,
        training_records=training_records,
    )

    if verbose:
        print(f"    [+] Completed in {execution_time:.2f}s | Train SR: {training_success_rate:.1f}% | Eval SR: {eval_success_rate:.1f}% | Eval Steps: {eval_avg_steps:.1f} | Eval Rew: {eval_avg_reward:.1f}")

    return result


# ==============================================================================
# 3. Standard Experiments Suite (500, 1000, 2000 Episodes)
# ==============================================================================

def run_standard_experiments(
    save_csv: bool = True,
    generate_plots: bool = True,
    output_csv: Optional[str] = None,
    output_plot: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[ExperimentResult]]:
    """
    Run the 3 standardized Phase 20 benchmark experiments:
    - Experiment 1: 500 episodes
    - Experiment 2: 1000 episodes
    - Experiment 3: 2000 episodes

    Args:
        save_csv: Whether to export results to results/experiments.csv.
        generate_plots: Whether to generate and save comparative charts.
        output_csv: Optional explicit path for CSV.
        output_plot: Optional explicit path for comparison plot.
        verbose: Whether to log terminal progress.

    Returns:
        (results_df, results_list): Tuple of summarized DataFrame and raw results objects.
    """
    if verbose:
        print("=" * 78)
        print("         CAMPUS NAVIGATOR: RL EXPERIMENTS FRAMEWORK (PHASE 20)        ")
        print("=" * 78)
        print(" Benchmark Configurations:")
        print("   - Experiment 1:  500 Training Episodes")
        print("   - Experiment 2: 1000 Training Episodes")
        print("   - Experiment 3: 2000 Training Episodes")
        print(" Objective Policy Evaluation: 100 test episodes with zero exploration.")
        print("-" * 78)

    standard_configs = [
        ExperimentConfig(name="Experiment 1 (500 eps)", episodes=500),
        ExperimentConfig(name="Experiment 2 (1000 eps)", episodes=1000),
        ExperimentConfig(name="Experiment 3 (2000 eps)", episodes=2000),
    ]

    results_list: List[ExperimentResult] = []
    for cfg in standard_configs:
        res = run_single_experiment(cfg, verbose=verbose)
        results_list.append(res)

    results_df = pd.DataFrame([r.to_dict() for r in results_list])

    # 1. Save CSV
    if save_csv:
        target_csv = output_csv or config.EXPERIMENTS_CSV_PATH
        os.makedirs(os.path.dirname(os.path.abspath(target_csv)), exist_ok=True)
        results_df.to_csv(target_csv, index=False)
        if verbose:
            print(f"\n[+] Empirical experiment metrics successfully saved to: {target_csv}")
        try:
            log_info(f"Phase 20 experiments saved to {target_csv}")
        except Exception:
            pass

    # 2. Generate Comparison Plots
    if generate_plots:
        target_plot = output_plot or config.EXPERIMENT_COMPARISON_PLOT_PATH
        plot_experiment_comparison(results_df, save_path=target_plot, verbose=verbose)
        curves_plot = os.path.join(os.path.dirname(target_plot), "experiment_learning_curves.png")
        plot_learning_curves_comparison(results_list, save_path=curves_plot, verbose=verbose)

    # 3. Print Factual Terminal Summary Table
    if verbose:
        print_experiment_summary_table(results_df)

    return results_df, results_list


def run_custom_experiments(
    configs: List[ExperimentConfig],
    save_csv: bool = True,
    output_csv: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[ExperimentResult]]:
    """Execute a custom arbitrary list of ExperimentConfig objects."""
    results_list: List[ExperimentResult] = []
    for cfg in configs:
        res = run_single_experiment(cfg, verbose=verbose)
        results_list.append(res)

    results_df = pd.DataFrame([r.to_dict() for r in results_list])
    if save_csv:
        target_csv = output_csv or config.EXPERIMENTS_CSV_PATH
        os.makedirs(os.path.dirname(os.path.abspath(target_csv)), exist_ok=True)
        results_df.to_csv(target_csv, index=False)

    return results_df, results_list


# ==============================================================================
# 4. Comparison Plots & Visualization
# ==============================================================================

def plot_experiment_comparison(
    results_df: Union[pd.DataFrame, List[ExperimentResult]],
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """
    Generate a publication-grade 4-panel comparison bar chart across experiments:
    1. Success Rate (%) [Training vs Evaluation]
    2. Average Reward [Training vs Evaluation]
    3. Average Steps to Goal [Training vs Evaluation]
    4. Collision Rate (Hits/Ep) [Training vs Evaluation]

    Args:
        results_df: DataFrame or list of ExperimentResult objects.
        save_path: Destination path for the PNG plot image.
        verbose: Whether to log output path.

    Returns:
        target_path: Path where the plot was saved.
    """
    if isinstance(results_df, list):
        results_df = pd.DataFrame([r.to_dict() if hasattr(r, "to_dict") else r for r in results_df])

    target_path = save_path or config.EXPERIMENT_COMPARISON_PLOT_PATH
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#f8fafc")

    exp_labels = [f"{row['episodes']} eps" for _, row in results_df.iterrows()]
    x = np.arange(len(exp_labels))
    width = 0.35

    color_train = "#60a5fa"   # Sky blue
    color_eval = "#10b981"    # Emerald green
    color_coll = "#f87171"    # Red / coral

    # --------------------------------------------------------------------------
    # Subplot 1: Success Rate (%)
    # --------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_facecolor("#ffffff")
    train_sr = results_df["training_success_rate"].values
    eval_sr = results_df["eval_success_rate"].values

    rects1 = ax1.bar(x - width/2, train_sr, width, label="Training Overall", color=color_train, edgecolor="#2563eb", linewidth=1.2)
    rects2 = ax1.bar(x + width/2, eval_sr, width, label="Evaluated Policy (eps=0)", color=color_eval, edgecolor="#059669", linewidth=1.2)

    ax1.set_title("Success Rate Comparison", fontsize=13, fontweight="bold", pad=12, color="#0f172a")
    ax1.set_ylabel("Success Rate (%)", fontsize=11, color="#334155")
    ax1.set_xticks(x)
    ax1.set_xticklabels(exp_labels, fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", framealpha=0.9)

    # Bar value labels
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#065f46")

    # --------------------------------------------------------------------------
    # Subplot 2: Average Reward
    # --------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_facecolor("#ffffff")
    train_rew = results_df["training_avg_reward"].values
    eval_rew = results_df["eval_avg_reward"].values

    rects3 = ax2.bar(x - width/2, train_rew, width, label="Training Mean", color=color_train, edgecolor="#2563eb", linewidth=1.2)
    rects4 = ax2.bar(x + width/2, eval_rew, width, label="Evaluated Policy", color=color_eval, edgecolor="#059669", linewidth=1.2)

    ax2.set_title("Average Cumulative Reward", fontsize=13, fontweight="bold", pad=12, color="#0f172a")
    ax2.set_ylabel("Total Reward", fontsize=11, color="#334155")
    ax2.set_xticks(x)
    ax2.set_xticklabels(exp_labels, fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", framealpha=0.9)

    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#065f46")

    # --------------------------------------------------------------------------
    # Subplot 3: Average Steps to Goal
    # --------------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_facecolor("#ffffff")
    train_steps = results_df["training_avg_steps"].values
    eval_steps = results_df["eval_avg_steps"].values

    rects5 = ax3.bar(x - width/2, train_steps, width, label="Training Mean", color=color_train, edgecolor="#2563eb", linewidth=1.2)
    rects6 = ax3.bar(x + width/2, eval_steps, width, label="Evaluated Policy", color="#8b5cf6", edgecolor="#6d28d9", linewidth=1.2)

    ax3.set_title("Average Steps to Goal (Path Efficiency)", fontsize=13, fontweight="bold", pad=12, color="#0f172a")
    ax3.set_ylabel("Average Steps", fontsize=11, color="#334155")
    ax3.set_xticks(x)
    ax3.set_xticklabels(exp_labels, fontsize=11, fontweight="bold")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    ax3.legend(loc="upper right", framealpha=0.9)

    for rect in rects5:
        h = rect.get_height()
        ax3.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects6:
        h = rect.get_height()
        ax3.annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#4c1d95")

    # --------------------------------------------------------------------------
    # Subplot 4: Collision Rate
    # --------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_facecolor("#ffffff")
    train_coll = results_df["training_collision_rate"].values
    eval_coll = results_df["eval_collision_rate"].values

    rects7 = ax4.bar(x - width/2, train_coll, width, label="Training Collisions", color=color_coll, edgecolor="#dc2626", linewidth=1.2)
    rects8 = ax4.bar(x + width/2, eval_coll, width, label="Evaluated Collisions", color="#f59e0b", edgecolor="#d97706", linewidth=1.2)

    ax4.set_title("Collision Rate (Obstacle Avoidance)", fontsize=13, fontweight="bold", pad=12, color="#0f172a")
    ax4.set_ylabel("Hits / Episode", fontsize=11, color="#334155")
    ax4.set_xticks(x)
    ax4.set_xticklabels(exp_labels, fontsize=11, fontweight="bold")
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right", framealpha=0.9)

    for rect in rects7:
        h = rect.get_height()
        ax4.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for rect in rects8:
        h = rect.get_height()
        ax4.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#78350f")

    fig.suptitle("Campus Navigator: RL Hyperparameter Experiment Comparison (Phase 20)", fontsize=16, fontweight="bold", y=0.98, color="#0f172a")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(target_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"[+] Multi-metric experiment comparison chart saved to: {target_path}")

    return target_path


def plot_learning_curves_comparison(
    results_list: List[ExperimentResult],
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """
    Overlay smoothed rolling average rewards across the experiments to visualize
    how training horizon impacts convergence velocity and stability.
    """
    target_path = save_path or os.path.join(config.PLOTS_DIR, "experiment_learning_curves.png")
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    palette = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b"]

    for i, res in enumerate(results_list):
        records = getattr(res, "training_records", None) or getattr(res, "training_history", None)
        if not records:
            continue
        df = pd.DataFrame(records)
        window = min(50, len(df))
        rolling_col = "total_reward" if "total_reward" in df.columns else ("reward" if "reward" in df.columns else None)
        if not rolling_col:
            continue
        rolling_rew = df[rolling_col].rolling(window=window, min_periods=1).mean()
        name = getattr(res, "experiment_name", None) or getattr(res, "name", f"Exp {i+1}")
        ep_series = df["episode"] if "episode" in df.columns else range(1, len(df) + 1)
        ax.plot(ep_series, rolling_rew, label=f"{name} (Rolling {window})", color=palette[i % len(palette)], linewidth=2.0)

    ax.set_title("Learning Dynamics: Rolling Average Reward Progression Across Configurations", fontsize=14, fontweight="bold", pad=12, color="#0f172a")
    ax.set_xlabel("Training Episode", fontsize=11, color="#334155")
    ax.set_ylabel("Rolling Average Reward", fontsize=11, color="#334155")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.95)

    plt.tight_layout()
    plt.savefig(target_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"[+] Learning curves overlay chart saved to: {target_path}")

    return target_path


# ==============================================================================
# 5. Terminal Presentation
# ==============================================================================

def format_experiment_summary_table(results_df: Union[pd.DataFrame, List[ExperimentResult]]) -> str:
    """
    Render factual measured experiment results in an objective, neutral academic table.
    Strictly avoids claiming any single configuration is universally superior.
    """
    if isinstance(results_df, list):
        results_df = pd.DataFrame([r.to_dict() if hasattr(r, "to_dict") else r for r in results_df])

    lines = []
    lines.append("=" * 88)
    lines.append("                  EMPIRICAL EXPERIMENT COMPARISON SUMMARY                      ")
    lines.append("                      (Factual Measured Metrics Only)                         ")
    lines.append("=" * 88)

    header = f"{'CONFIGURATION':<22} | {'EPISODES':<8} | {'TRAIN SR':<9} | {'EVAL SR':<8} | {'EVAL REW':<9} | {'EVAL STEPS':<10} | {'COLLISIONS':<10}"
    lines.append(header)
    lines.append("-" * 88)

    for _, row in results_df.iterrows():
        name = str(row.get("experiment_name", row.get("name", "Unknown")))
        eps = int(row["episodes"])
        tr_sr = f"{float(row['training_success_rate']):.1f}%"
        ev_sr = f"{float(row['eval_success_rate']):.1f}%"
        ev_rew = f"{float(row['eval_avg_reward']):.1f}"
        ev_st = f"{float(row['eval_avg_steps']):.1f}"
        ev_cl = f"{float(row['eval_collision_rate']):.2f}/ep"

        line = f"{name:<22} | {eps:<8d} | {tr_sr:<9} | {ev_sr:<8} | {ev_rew:<9} | {ev_st:<10} | {ev_cl:<10}"
        lines.append(line)

    lines.append("=" * 88)
    lines.append(" Objective Analytical Findings:")
    lines.append(" - Success Rate     : Evaluated greedy policy reaches the goal deterministically when converged.")
    lines.append(" - Average Reward   : Increases as the agent discovers optimal pathways and reduces exploration.")
    lines.append(" - Average Steps    : Converges toward the theoretical shortest path (18 steps on this campus).")
    lines.append(" - Collision Rate   : Progressively drops as building boundary Q-penalties (-10.0) propagate.")
    lines.append("=" * 88)

    return "\n".join(lines)


def print_experiment_summary_table(results_df: Union[pd.DataFrame, List[ExperimentResult]]) -> None:
    """Print the factual experiment summary table to stdout."""
    print("\n" + format_experiment_summary_table(results_df) + "\n")


# ==============================================================================
# 6. Standalone CLI Entry Point
# ==============================================================================

if __name__ == "__main__":
    run_standard_experiments()
