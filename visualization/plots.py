"""
Campus Navigator: Training Visualization (Phase 5)
=================================================
Module: visualization/plots.py

RL Concept - PERFORMANCE EVALUATION THROUGH VISUALIZATION:
In Reinforcement Learning, evaluating agent learning requires tracking
key diagnostic metrics across training episodes:
1. Reward vs Episode: Tracks learning progression and convergence to optimal return.
2. Steps vs Episode: Illustrates path efficiency as navigation time decreases.
3. Success Rate vs Episode: Rolling window percentage of successful arrivals.
4. Collision Rate vs Episode: Demonstrates obstacle and boundary avoidance learning.
5. Moving Average Reward: Smoothed trendline eliminating stochastic exploration noise.

All charts are generated strictly from actual training data in:
results/training_results.csv
and saved into:
results/plots/
"""

import os
import sys
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is accessible when running standalone
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config


def load_training_data(csv_path: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Safely load and validate the training metrics CSV.

    Args:
        csv_path: Optional explicit path to the CSV file.

    Returns:
        (df, error_message): Tuple of loaded DataFrame (or None) and error string (or None).
    """
    target_path = csv_path or getattr(config, "TRAINING_RESULTS_PATH", config.METRICS_CSV_PATH)

    if not os.path.exists(target_path):
        err = f"[!] Error: Training results file not found at '{target_path}'.\n    Please run training first using: python training/train.py"
        try:
            from utils.logger import log_warning
            log_warning(f"Training results CSV not found at '{target_path}'.")
        except ImportError:
            pass
        return None, err

    if os.path.getsize(target_path) == 0:
        err = f"[!] Error: Training results file at '{target_path}' is completely empty (0 bytes).\n    Please run training to record metrics."
        try:
            from utils.logger import log_warning
            log_warning(f"Training results CSV '{target_path}' is empty (0 bytes).")
        except ImportError:
            pass
        return None, err

    try:
        df = pd.read_csv(target_path)
    except pd.errors.EmptyDataError:
        err = f"[!] Error: Training results file at '{target_path}' contains no data rows.\n    Please run training to generate metrics."
        try:
            from utils.logger import log_warning
            log_warning(f"EmptyDataError encountered reading '{target_path}'.")
        except ImportError:
            pass
        return None, err
    except Exception as e:
        err = f"[!] Error reading '{target_path}': {e}"
        try:
            from utils.logger import log_exception
            log_exception(e, f"load_training_data({target_path})")
        except ImportError:
            pass
        return None, err

    if len(df) == 0:
        err = f"[!] Error: Training results dataset in '{target_path}' is empty (0 rows).\n    Please run training to generate performance records."
        try:
            from utils.logger import log_warning
            log_warning(f"Training results CSV '{target_path}' has 0 rows.")
        except ImportError:
            pass
        return None, err

    if len(df) < 2:
        err = f"[!] Error: Not enough training data in '{target_path}' (found {len(df)} episodes, need at least 2).\n    Please run training for more episodes."
        try:
            from utils.logger import log_warning
            log_warning(f"Insufficient training episodes in '{target_path}': {len(df)} found, >=2 needed.")
        except ImportError:
            pass
        return None, err

    required_cols = {"episode", "total_reward", "steps"}
    if not required_cols.issubset(df.columns):
        err = f"[!] Error: Missing required columns in '{target_path}'. Expected at least {required_cols}, found {list(df.columns)}."
        try:
            from utils.logger import log_error
            log_error(err)
        except ImportError:
            pass
        return None, err

    return df, None


# ------------------------------------------------------------------------------
# 1. Reward vs Episode
# ------------------------------------------------------------------------------

def plot_reward_vs_episode(
    df: pd.DataFrame,
    output_dir: str,
    window_size: int = 50,
) -> str:
    """
    Plot raw episode rewards alongside a moving average trendline.
    Saves to: results/plots/reward_vs_episode.png
    """
    plt.figure(figsize=(10, 5.5))
    plt.plot(
        df["episode"],
        df["total_reward"],
        color="#20c997",
        alpha=0.35,
        label="Raw Episode Reward",
        linewidth=1.0,
    )

    actual_window = min(window_size, max(2, len(df) // 4))
    rolling_mean = df["total_reward"].rolling(window=actual_window, min_periods=1).mean()
    plt.plot(
        df["episode"],
        rolling_mean,
        color="#0d6efd",
        linewidth=2.2,
        label=f"{actual_window}-Episode Moving Average",
    )

    plt.title("Campus Navigator: Total Reward vs Episode", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Episode Number", fontsize=12)
    plt.ylabel("Cumulative Episode Reward", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "reward_vs_episode.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


# ------------------------------------------------------------------------------
# 2. Steps vs Episode
# ------------------------------------------------------------------------------

def plot_steps_vs_episode(
    df: pd.DataFrame,
    output_dir: str,
    window_size: int = 50,
) -> str:
    """
    Plot navigation step count per episode demonstrating path efficiency convergence.
    Saves to: results/plots/steps_vs_episode.png
    """
    plt.figure(figsize=(10, 5.5))
    plt.plot(
        df["episode"],
        df["steps"],
        color="#fd7e14",
        alpha=0.35,
        label="Steps per Episode",
        linewidth=1.0,
    )

    actual_window = min(window_size, max(2, len(df) // 4))
    rolling_steps = df["steps"].rolling(window=actual_window, min_periods=1).mean()
    plt.plot(
        df["episode"],
        rolling_steps,
        color="#dc3545",
        linewidth=2.2,
        label=f"{actual_window}-Episode Moving Average",
    )

    # Theoretical minimum Manhattan distance benchmark (18 steps)
    plt.axhline(
        y=18,
        color="#198754",
        linestyle="--",
        linewidth=1.8,
        label="Optimal Theoretical Path (18 Steps)",
    )

    plt.title("Campus Navigator: Steps vs Episode (Path Efficiency)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Episode Number", fontsize=12)
    plt.ylabel("Steps to Reach Destination", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "steps_vs_episode.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


# ------------------------------------------------------------------------------
# 3. Success Rate vs Episode
# ------------------------------------------------------------------------------

def plot_success_rate(
    df: pd.DataFrame,
    output_dir: str,
    window_size: int = 50,
) -> str:
    """
    Plot the rolling window success rate percentage across training episodes.
    Saves to: results/plots/success_rate.png
    """
    plt.figure(figsize=(10, 5.5))

    actual_window = min(window_size, max(2, len(df) // 4))
    success_col = df["success"] if "success" in df.columns else (df["total_reward"] > 0).astype(int)

    # Calculate rolling percentage
    rolling_success_pct = (
        success_col.rolling(window=actual_window, min_periods=1).mean() * 100.0
    )
    # Calculate cumulative percentage
    cumulative_success_pct = (
        success_col.cumsum() / (df.index + 1) * 100.0
    )

    plt.plot(
        df["episode"],
        rolling_success_pct,
        color="#0d6efd",
        linewidth=2.2,
        label=f"{actual_window}-Episode Rolling Success Rate",
    )
    plt.plot(
        df["episode"],
        cumulative_success_pct,
        color="#6c757d",
        linestyle=":",
        linewidth=1.8,
        label="Cumulative Overall Success Rate",
    )

    plt.title("Campus Navigator: Success Rate vs Episode", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Episode Number", fontsize=12)
    plt.ylabel("Arrival Success Rate (%)", fontsize=12)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "success_rate.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


# ------------------------------------------------------------------------------
# 4. Collision Rate vs Episode
# ------------------------------------------------------------------------------

def plot_collision_rate(
    df: pd.DataFrame,
    output_dir: str,
    window_size: int = 50,
) -> str:
    """
    Plot collisions (wall boundary & obstacle collisions) per episode and rolling rate.
    Saves to: results/plots/collision_rate.png
    """
    plt.figure(figsize=(10, 5.5))

    if "collisions" in df.columns:
        collisions_series = df["collisions"]
    else:
        collisions_series = pd.Series([0] * len(df))

    plt.plot(
        df["episode"],
        collisions_series,
        color="#e83e8c",
        alpha=0.3,
        label="Collisions per Episode",
        linewidth=1.0,
    )

    actual_window = min(window_size, max(2, len(df) // 4))
    rolling_collisions = collisions_series.rolling(window=actual_window, min_periods=1).mean()
    plt.plot(
        df["episode"],
        rolling_collisions,
        color="#6f42c1",
        linewidth=2.2,
        label=f"{actual_window}-Episode Rolling Collision Rate",
    )

    plt.title("Campus Navigator: Collision Rate vs Episode (Obstacle Avoidance)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Episode Number", fontsize=12)
    plt.ylabel("Number of Collisions", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "collision_rate.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


# ------------------------------------------------------------------------------
# 5. Moving Average Reward
# ------------------------------------------------------------------------------

def plot_moving_average_reward(
    df: pd.DataFrame,
    output_dir: str,
    window_size: int = 50,
) -> str:
    """
    Dedicated convergence analysis graph: Moving Average Reward with rolling std deviation band.
    Saves to: results/plots/moving_average_reward.png
    """
    plt.figure(figsize=(10, 5.5))

    actual_window = min(window_size, max(2, len(df) // 4))
    rolling_mean = df["total_reward"].rolling(window=actual_window, min_periods=1).mean()
    rolling_std = df["total_reward"].rolling(window=actual_window, min_periods=1).std().fillna(0)

    # Shaded variance band (confidence interval)
    plt.fill_between(
        df["episode"],
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        color="#0d6efd",
        alpha=0.18,
        label=f"+/- 1 Std Dev ({actual_window}-Ep Window)",
    )

    # Main smoothed reward line
    plt.plot(
        df["episode"],
        rolling_mean,
        color="#0d6efd",
        linewidth=2.5,
        label=f"{actual_window}-Episode Moving Average Reward",
    )

    # Zero reward reference line
    plt.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

    plt.title("Campus Navigator: Moving Average Reward (Policy Convergence)", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Episode Number", fontsize=12)
    plt.ylabel("Smoothed Cumulative Reward", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "moving_average_reward.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


# ------------------------------------------------------------------------------
# Master Dispatcher: generate_all_plots()
# ------------------------------------------------------------------------------

def generate_all_plots(
    csv_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    """
    Load actual training data and generate all 5 required Matplotlib plots.

    Files generated in results/plots/:
    - reward_vs_episode.png
    - steps_vs_episode.png
    - success_rate.png
    - collision_rate.png
    - moving_average_reward.png

    Args:
        csv_path: Optional path to training_results.csv. Defaults to config.TRAINING_RESULTS_PATH.
        output_dir: Target directory. Defaults to config.PLOTS_DIR (results/plots/).
        verbose: If True, prints progress and generated file locations.

    Returns:
        saved_plots: Dict mapping graph names to their absolute filesystem paths.
    """
    # 1. Load & Validate Data
    df, err = load_training_data(csv_path)
    if err is not None:
        if verbose:
            print(err)
        return {}

    # 2. Prepare Output Directory
    target_dir = output_dir or getattr(config, "PLOTS_DIR", os.path.join(config.RESULTS_DIR, "plots"))
    os.makedirs(target_dir, exist_ok=True)

    if verbose:
        print("=" * 76)
        print("        CAMPUS NAVIGATOR: GENERATING TRAINING PERFORMANCE PLOTS       ")
        print("=" * 76)
        print(f" Source Data    : {len(df)} episodes loaded from CSV")
        print(f" Target Folder  : {target_dir}")
        print("-" * 76)

    # 3. Generate Each Required Plot
    saved_plots = {}

    p1 = plot_reward_vs_episode(df, target_dir)
    saved_plots["reward_vs_episode"] = p1
    if verbose:
        print(f" [1/5] Generated : {os.path.basename(p1)}")

    p2 = plot_steps_vs_episode(df, target_dir)
    saved_plots["steps_vs_episode"] = p2
    if verbose:
        print(f" [2/5] Generated : {os.path.basename(p2)}")

    p3 = plot_success_rate(df, target_dir)
    saved_plots["success_rate"] = p3
    if verbose:
        print(f" [3/5] Generated : {os.path.basename(p3)}")

    p4 = plot_collision_rate(df, target_dir)
    saved_plots["collision_rate"] = p4
    if verbose:
        print(f" [4/5] Generated : {os.path.basename(p4)}")

    p5 = plot_moving_average_reward(df, target_dir)
    saved_plots["moving_average_reward"] = p5
    if verbose:
        print(f" [5/5] Generated : {os.path.basename(p5)}")

    if verbose:
        print("-" * 76)
        print(f"[+] All 5 professional Matplotlib graphs successfully saved to:\n    {target_dir}")
        print("=" * 76 + "\n")

    return saved_plots


# Backward compatibility aliases
plot_training_rewards = plot_reward_vs_episode
plot_training_steps = plot_steps_vs_episode


def plot_comprehensive_dashboard(
    metrics_df: Optional[pd.DataFrame] = None,
    window_size: int = 50,
    save: bool = True,
    show: bool = False,
) -> str:
    """
    Generate a 3-panel educational performance dashboard for project viva demonstration.
    """
    df, err = load_training_data() if metrics_df is None else (metrics_df, None)
    if err is not None:
        print(err)
        return ""

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    actual_window = min(window_size, max(2, len(df) // 4))

    # 1. Total Rewards
    axes[0].plot(df["episode"], df["total_reward"], alpha=0.35, color="teal", label="Raw Episode Reward")
    rolling_mean = df["total_reward"].rolling(window=actual_window, min_periods=1).mean()
    axes[0].plot(df["episode"], rolling_mean, color="navy", linewidth=2.0, label=f"{actual_window}-Ep Moving Avg")
    axes[0].set_ylabel("Total Reward", fontsize=11)
    axes[0].set_title("Training Reward Curve (Convergence to Optimal Policy)", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="lower right")

    # 2. Steps to Destination
    axes[1].plot(df["episode"], df["steps"], alpha=0.35, color="coral", label="Steps per Episode")
    rolling_steps = df["steps"].rolling(window=actual_window, min_periods=1).mean()
    axes[1].plot(df["episode"], rolling_steps, color="darkred", linewidth=2.0, label=f"{actual_window}-Ep Moving Avg")
    axes[1].axhline(18, color="green", linestyle="--", label="Optimal Shortest Path (18 Steps)")
    axes[1].set_ylabel("Steps Taken", fontsize=11)
    axes[1].set_title("Step Count per Episode (Navigation Efficiency)", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right")

    # 3. Success Rate
    success_col = df["success"] if "success" in df.columns else (df["total_reward"] > 0).astype(int)
    rolling_success = success_col.rolling(window=actual_window, min_periods=1).mean() * 100.0
    axes[2].plot(df["episode"], rolling_success, color="purple", linewidth=2.0, label=f"{actual_window}-Ep Rolling Success Rate (%)")
    axes[2].set_ylabel("Success Rate (%)", fontsize=11)
    axes[2].set_xlabel("Episode Number", fontsize=11)
    axes[2].set_ylim(-5, 105)
    axes[2].set_title("Arrival Success Rate Across Training", fontsize=12, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="lower right")

    plt.tight_layout()

    save_path = os.path.join(config.RESULTS_DIR, "performance_dashboard.png")
    if save:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close()
    return save_path


if __name__ == "__main__":
    generate_all_plots()
