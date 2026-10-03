"""
Campus Navigator: Policy Evaluation (Phase 4)
=============================================
Module: training/evaluate.py

RL Concept - POLICY EVALUATION & GREEDY EXPLOITATION:
In Reinforcement Learning, after training is complete, the learned policy
pi*(s) = argmax_a Q(s, a) must be evaluated without exploration:
    1. Set exploration rate epsilon = 0.0 (pure exploitation).
    2. The agent selects actions purely based on the learned Q-table.
    3. The Q-table is NOT updated during evaluation (weights are frozen).
    4. Evaluates performance across at least 100 episodes to measure:
       - Success Rate (% of episodes arriving at Central Library)
       - Average Cumulative Reward
       - Average Steps to Goal
       - Minimum & Maximum Steps
       - Collision Rate (average collisions per episode)
       - Average Route Length
       - The exact learned optimal path coordinates.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Ensure project root is accessible when running standalone
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config
from environment.campus_env import CampusEnvironment
from agent.q_learning import QLearningAgent


def evaluate_agent(
    episodes: int = 100,
    max_steps: int = config.MAX_STEPS_PER_EPISODE,
    model_path: Optional[str] = None,
    save_csv: bool = True,
    save_json: bool = True,
    verbose: bool = True,
    agent: Optional[QLearningAgent] = None,
    env: Optional[CampusEnvironment] = None,
    eval_episodes: Optional[int] = None,
) -> Dict:
    """
    Evaluate the learned policy of the Q-Learning agent without exploration.

    This function is designed to be imported by the game UI or run standalone.
    It guarantees that the Q-table is FROZEN and NEVER modified.

    Args:
        episodes: Number of evaluation episodes (default: 100).
        max_steps: Maximum step horizon per episode (default: 200).
        model_path: Path to the saved Q-table .npy file.
        save_csv: Whether to save per-episode data to results/evaluation_results.csv.
        save_json: Whether to save aggregated summary to results/evaluation_summary.json.
        verbose: Whether to print the summary to the terminal.
        agent: Optional pre-loaded agent instance.
        env: Optional pre-existing CampusEnvironment instance.
        eval_episodes: Optional alias for episodes.

    Returns:
        summary: Dictionary containing comprehensive evaluation metrics and learned path.
    """
    if eval_episodes is not None:
        episodes = eval_episodes
    # 1. Initialize Environment
    if env is None:
        env = CampusEnvironment(
            rows=config.GRID_ROWS,
            cols=config.GRID_COLS,
            max_steps=max_steps,
        )
    else:
        env.max_steps = max_steps

    # 2. Initialize and Load Agent
    if agent is None:
        agent = QLearningAgent(
            rows=env.rows,
            cols=env.cols,
            num_actions=config.NUM_ACTIONS,
        )
        target_model_path = model_path or config.Q_TABLE_SAVE_PATH
        if os.path.exists(target_model_path):
            loaded = agent.load_model(target_model_path)
            if verbose:
                print(f"[+] Loaded trained Q-table from: {target_model_path}")
        else:
            if verbose:
                print(f"[!] Warning: No model found at '{target_model_path}'. Evaluating untrained agent.")
    elif model_path is not None and os.path.exists(model_path):
        agent.load_model(model_path)
        if verbose:
            print(f"[+] Loaded trained Q-table into existing agent from: {model_path}")

    # 3. CRITICAL: Freeze Exploration (Pure Greedy Policy)
    agent.epsilon = 0.0

    records = []
    successful_paths: List[List[Tuple[int, int]]] = []
    total_rewards: List[float] = []
    total_steps: List[int] = []
    collisions_list: List[int] = []

    if verbose:
        print("=" * 76)
        print("     CAMPUS NAVIGATOR: POLICY EVALUATION (PHASE 4 - ZERO EPSILON)    ")
        print("=" * 76)
        print(f" Evaluation Episodes : {episodes}")
        print(f" Max Steps / Episode : {max_steps}")
        print(f" Exploration Epsilon : {agent.epsilon} (Pure Exploitation)")
        print(f" Q-Table Update      : DISABLED (Weights Frozen)")
        print("-" * 76)

    # --------------------------------------------------------------------------
    # Evaluation Loop
    # --------------------------------------------------------------------------
    for ep in range(1, episodes + 1):
        state = env.reset()
        ep_reward: float = 0.0
        steps: int = 0
        collisions: int = 0
        done: bool = False
        success: bool = False

        # Track full coordinate trajectory
        path: List[Tuple[int, int]] = [state]

        while not done:
            # Select action with PURE GREEDY EXPLOITATION (no exploration)
            action = agent.get_best_action(state)

            # Step in environment WITHOUT agent.update()
            next_state, reward, done, info = env.step(action)

            # Track collision events
            if not info.get("is_valid", True):
                collisions += 1

            if info.get("goal_reached", False):
                success = True

            state = next_state
            ep_reward += reward
            steps += 1
            path.append(state)

        total_rewards.append(ep_reward)
        total_steps.append(steps)
        collisions_list.append(collisions)

        if success:
            successful_paths.append(path)

        records.append({
            "episode": ep,
            "total_reward": round(ep_reward, 2),
            "steps": steps,
            "success": 1 if success else 0,
            "collisions": collisions,
            "route_length": len(path),
        })

    eval_df = pd.DataFrame(records)

    # --------------------------------------------------------------------------
    # Compute Evaluation Metrics
    # --------------------------------------------------------------------------
    total_eval_episodes = len(eval_df)
    successful_episodes = int(eval_df["success"].sum())
    success_rate = (successful_episodes / total_eval_episodes) * 100.0
    avg_reward = float(np.mean(total_rewards))
    avg_steps = float(np.mean(total_steps))
    min_steps = int(np.min(total_steps)) if len(total_steps) > 0 else 0
    max_steps_taken = int(np.max(total_steps)) if len(total_steps) > 0 else 0
    collision_rate = float(np.mean(collisions_list))
    total_collisions_count = int(np.sum(collisions_list))

    # Route Lengths (number of coordinates in path)
    if successful_paths:
        avg_route_length = float(np.mean([len(p) for p in successful_paths]))
        sample_learned_path = successful_paths[0]
    else:
        avg_route_length = float(np.mean(eval_df["route_length"]))
        sample_learned_path = path

    learned_path_coords = [list(coord) for coord in sample_learned_path]

    summary = {
        "evaluation_episodes": total_eval_episodes,
        "episodes": total_eval_episodes,
        "successful_episodes": successful_episodes,
        "success_rate_percent": round(success_rate, 2),
        "success_rate": round(success_rate, 2),
        "average_reward": round(avg_reward, 2),
        "avg_reward": round(avg_reward, 2),
        "average_steps": round(avg_steps, 2),
        "avg_steps": round(avg_steps, 2),
        "min_steps": min_steps,
        "max_steps": max_steps_taken,
        "total_collisions": total_collisions_count,
        "collision_rate": round(collision_rate, 4),
        "average_route_length": round(avg_route_length, 2),
        "learned_path_length": round(avg_route_length, 2),
        "learned_path": learned_path_coords,
        "optimal_path": learned_path_coords,
    }

    # --------------------------------------------------------------------------
    # Save Outputs
    # --------------------------------------------------------------------------
    if save_csv:
        csv_path = config.EVAL_CSV_PATH
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        eval_df.to_csv(csv_path, index=False)
        if verbose:
            print(f"[+] Evaluation per-episode results saved to: {csv_path}")

    if save_json:
        json_path = config.EVAL_SUMMARY_PATH
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=4)
        if verbose:
            print(f"[+] Evaluation summary metrics saved to: {json_path}")

    # --------------------------------------------------------------------------
    # Print Clean Summary
    # --------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 76)
        print("                  EVALUATION RUN SUMMARY (FROZEN POLICY)                ")
        print("=" * 76)
        print(f"  Total Evaluation Episodes  : {total_eval_episodes}")
        print(f"  Successful Arrivals        : {successful_episodes} / {total_eval_episodes}")
        print(f"  1. Success Rate            : {summary['success_rate_percent']:.2f}%")
        print(f"  2. Average Cumulative Reward: {summary['average_reward']:.2f}")
        print(f"  3. Average Steps to Goal   : {summary['average_steps']:.2f}")
        print(f"  4. Minimum Steps Recorded  : {summary['min_steps']}")
        print(f"  5. Maximum Steps Recorded  : {summary['max_steps']}")
        print(f"  6. Collision Rate          : {summary['collision_rate']:.4f} collisions/ep (Total: {total_collisions_count})")
        print(f"  7. Average Route Length    : {summary['average_route_length']:.2f} coordinates")
        print("-" * 76)
        print(f"  Learned Optimal Route ({len(sample_learned_path)} coordinates, {len(sample_learned_path)-1} steps):")
        formatted_path = " -> ".join([str(c) for c in sample_learned_path])
        print(f"  {formatted_path}")
        print("=" * 76 + "\n")

    return summary


# Alias for backward compatibility across the codebase
evaluate_policy = evaluate_agent


def evaluate_multi_destinations(
    episodes_per_dest: int = 20,
    max_steps: int = config.MAX_STEPS_PER_EPISODE,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Evaluate policy performance across all 5 predefined campus destinations (Phase 16).

    Returns:
        multi_summary: Dictionary mapping destination names to their performance summaries.
    """
    env = CampusEnvironment(rows=config.GRID_ROWS, cols=config.GRID_COLS, max_steps=max_steps)
    agent = QLearningAgent(rows=env.rows, cols=env.cols, num_actions=config.NUM_ACTIONS)
    agent.load_model()
    agent.epsilon = 0.0

    multi_summary = {}
    if verbose:
        print("=" * 76)
        print("     CAMPUS NAVIGATOR: MULTI-DESTINATION EVALUATION (PHASE 16)      ")
        print("=" * 76)

    start_locations = list(config.START_LOCATIONS.values())

    for dest_idx, (dest_name, dest_coord) in enumerate(config.DESTINATIONS.items()):
        agent.set_destination_id(dest_idx)
        env.set_destination(dest_coord)

        successes = 0
        total_steps = []
        total_rewards = []

        for ep in range(episodes_per_dest):
            s_idx = ep % len(start_locations)
            s_pos = start_locations[s_idx]
            if s_pos == dest_coord:
                s_pos = (0, 0)
            state = env.reset(start_pos=s_pos, goal_pos=dest_coord)
            ep_reward = 0.0
            steps = 0

            for _ in range(max_steps):
                action = agent.get_best_action(state)
                next_state, reward, done, info = env.step(action)
                ep_reward += reward
                steps += 1
                state = next_state
                if done:
                    if info.get("goal_reached", False):
                        successes += 1
                    break

            total_steps.append(steps)
            total_rewards.append(ep_reward)

        sr = (successes / episodes_per_dest) * 100.0
        avg_s = float(np.mean(total_steps))
        avg_r = float(np.mean(total_rewards))
        multi_summary[dest_name] = {
            "destination": dest_name,
            "coordinate": dest_coord,
            "success_rate": round(sr, 1),
            "avg_steps": round(avg_s, 1),
            "avg_reward": round(avg_r, 1),
        }
        if verbose:
            print(f"  Destination [{dest_idx}] {dest_name:<15} ({dest_coord}): Success={sr:5.1f}% | Avg Steps={avg_s:4.1f} | Avg Reward={avg_r:+6.1f}")

    if verbose:
        print("=" * 76 + "\n")

    return multi_summary


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for evaluation execution."""
    parser = argparse.ArgumentParser(
        description="Campus Navigator: Policy Evaluation (Phase 4 & 16)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=config.MAX_STEPS_PER_EPISODE,
        help="Maximum steps allowed per episode (default: 200).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.Q_TABLE_SAVE_PATH,
        help="Path to trained Q-table .npy file.",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Evaluate performance across all 5 predefined campus destinations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.multi:
        evaluate_multi_destinations()
    else:
        evaluate_agent(
            episodes=args.episodes,
            max_steps=args.max_steps,
            model_path=args.model,
        )
