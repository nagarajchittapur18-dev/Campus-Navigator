"""
Campus Navigator: Training Loop (Phase 3)
=========================================
Module: training/train.py

RL Concept - TABULAR Q-LEARNING TRAINING PROCESS:
Connects the CampusEnvironment and the QLearningAgent to facilitate
experiential reinforcement learning across multiple episodes.

Algorithm per Episode:
----------------------
1. Reset the environment to the starting state S_0 (Campus Main Gate).
2. For step t = 1 to max_steps:
     a. Select action A_t using epsilon-greedy policy:
        - With probability epsilon: explore random action.
        - With probability (1 - epsilon): exploit best learned Q-value.
     b. Execute action A_t in environment.
     c. Receive immediate reward R_{t+1}, next state S_{t+1}, and terminal flag.
     d. Update Q(S_t, A_t) using the Bellman Optimality Equation:
        Q(S_t, A_t) <- Q(S_t, A_t) + alpha * [ R_{t+1} + gamma * max_a Q(S_{t+1}, a) - Q(S_t, A_t) ]
     e. Check if move was invalid to record collisions.
     f. S_t <- S_{t+1}.
     g. If terminal state reached (Goal or max steps): break episode.
3. Record episode statistics (reward, steps, success, collisions, epsilon).
4. Decay exploration rate epsilon: epsilon <- max(epsilon_min, epsilon * epsilon_decay).
5. Periodically log progress and persist models and metrics.
"""

import argparse
import os
import sys
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Ensure project root is in sys.path when running standalone
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config
from environment.campus_env import CampusEnvironment
from agent.q_learning import QLearningAgent


def train_agent(
    episodes: int = config.DEFAULT_TRAINING_EPISODES,
    max_steps: int = config.MAX_STEPS_PER_EPISODE,
    env: Optional[CampusEnvironment] = None,
    agent: Optional[QLearningAgent] = None,
    save_model: bool = True,
    save_results: bool = True,
    log_interval: int = 100,
    verbose: bool = True,
    model_path: Optional[str] = None,
    results_path: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> Tuple[QLearningAgent, pd.DataFrame]:
    """
    Train a Tabular Q-Learning Agent in the Campus Environment.

    Args:
        episodes: Total number of training episodes (default: 1000).
        max_steps: Maximum step horizon per episode (default: 200).
        env: Optional pre-existing CampusEnvironment instance.
        agent: Optional pre-existing QLearningAgent instance.
        save_model: Whether to save the trained Q-table to disk.
        save_results: Whether to save metrics to CSV.
        log_interval: Episode frequency to display progress in terminal.
        verbose: Whether to print progress and end-of-training summary.
        model_path: Optional explicit file path for saving the Q-table .npy file.
        results_path: Optional explicit file path for saving training_results.csv.
        random_seed: Optional random seed for reproducible exploration trajectories.

    Returns:
        agent: Trained QLearningAgent with updated Q-table.
        results_df: Pandas DataFrame containing logged episode records.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if env is None:
        env = CampusEnvironment(
            rows=config.GRID_ROWS,
            cols=config.GRID_COLS,
            max_steps=max_steps,
        )
    else:
        env.max_steps = max_steps

    if agent is None:
        agent = QLearningAgent(
            rows=env.rows,
            cols=env.cols,
            num_actions=config.NUM_ACTIONS,
            learning_rate=config.LEARNING_RATE,
            discount_factor=config.DISCOUNT_FACTOR,
            epsilon=config.EPSILON_INITIAL,
            epsilon_min=config.EPSILON_MIN,
            epsilon_decay=config.EPSILON_DECAY,
        )

    records = []
    initial_non_zeros = int(np.count_nonzero(agent.q_table))

    if verbose:
        print("=" * 76)
        print("     CAMPUS NAVIGATOR: Q-LEARNING AGENT TRAINING PROCESS (PHASE 3)   ")
        print("=" * 76)
        print(f" Grid Environment    : {env.rows}x{env.cols} Campus Grid")
        print(f" Training Episodes   : {episodes}")
        print(f" Max Steps / Episode : {max_steps}")
        print(f" Hyperparameters     : alpha={agent.alpha}, gamma={agent.gamma}, eps={agent.epsilon} (min={agent.epsilon_min})")
        print(f" Initial Non-Zero Q  : {initial_non_zeros} / {agent.q_table.size}")
        print("-" * 76)

    # --------------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------------
    # Main Training Loop (with Graceful Interruption Handling - Phase 18)
    # --------------------------------------------------------------------------
    try:
        for ep in range(1, episodes + 1):
            # 1. Reset environment
            state = env.reset()

            total_reward: float = 0.0
            steps: int = 0
            collisions: int = 0
            done: bool = False
            success: bool = False

            current_epsilon = agent.epsilon

            # Episode step horizon
            while not done:
                # 2. Select action using epsilon-greedy
                action = agent.choose_action(state, training=True)

                # 3. Execute action in environment
                next_state, reward, done, info = env.step(action)

                # 4. Update Q-table via Bellman Optimality Equation
                agent.update(state, action, reward, next_state, done)

                # 5. Track collisions (wall bounds or obstacle hits)
                if not info.get("is_valid", True):
                    collisions += 1

                if info.get("goal_reached", False):
                    success = True

                state = next_state
                total_reward += reward
                steps += 1

            # 6. Decay exploration rate after episode completion
            agent.decay_epsilon()

            # 7. Record episode statistics
            records.append({
                "episode": ep,
                "total_reward": round(total_reward, 2),
                "steps": steps,
                "success": 1 if success else 0,
                "collisions": collisions,
                "epsilon": round(current_epsilon, 4),
            })

            # 8. Periodic Progress Logging
            if verbose and (ep % log_interval == 0 or ep == 1 or ep == episodes):
                print(
                    f"  Episode {ep:4d}/{episodes} | "
                    f"Reward: {total_reward:7.1f} | "
                    f"Steps: {steps:3d} | "
                    f"Collisions: {collisions:2d} | "
                    f"Epsilon: {current_epsilon:.4f} | "
                    f"Success: {'YES' if success else 'NO'}"
                )
    except KeyboardInterrupt:
        if verbose:
            print(f"\n[!] Training interrupted by user at episode {len(records)}/{episodes}.")
            print("[+] Safely preserving learned Q-values and recorded metrics to disk...")
        try:
            from utils.logger import log_warning
            log_warning(f"Training loop interrupted by user via KeyboardInterrupt at episode {len(records)}/{episodes}. Preserving progress.")
        except ImportError:
            pass

    results_cols = ["episode", "steps", "total_reward", "success", "collisions", "epsilon"]
    results_df = pd.DataFrame(records, columns=results_cols)

    # --------------------------------------------------------------------------
    # Persistence
    # --------------------------------------------------------------------------
    if save_model:
        target_model = model_path or config.Q_TABLE_SAVE_PATH
        saved_model_path = agent.save_model(target_model)
        if verbose:
            print("-" * 76)
            print(f"[+] Trained Q-Table saved successfully to: {saved_model_path}")

    if save_results:
        target_results = results_path or config.TRAINING_RESULTS_PATH
        os.makedirs(os.path.dirname(os.path.abspath(target_results)), exist_ok=True)
        results_df.to_csv(target_results, index=False)
        if verbose:
            print(f"[+] Training statistics saved to CSV: {target_results}")

    # --------------------------------------------------------------------------
    # Final Summary Statistics
    # --------------------------------------------------------------------------
    total_episodes = len(results_df)
    if total_episodes == 0:
        if verbose:
            print("\n[!] Training ended before completing any full episodes.")
        return agent, results_df

    successful_episodes = int(results_df["success"].sum())
    success_rate = (successful_episodes / total_episodes) * 100.0
    avg_reward = float(results_df["total_reward"].mean())
    avg_steps = float(results_df["steps"].mean())
    total_collisions = int(results_df["collisions"].sum())
    avg_collisions = float(results_df["collisions"].mean())
    final_epsilon = float(agent.epsilon)
    final_non_zeros = int(np.count_nonzero(agent.q_table))

    # Calculate last 100 episodes performance (converged performance)
    last_100_df = results_df.tail(100)
    last_100_success_rate = (last_100_df["success"].sum() / len(last_100_df)) * 100.0
    last_100_avg_steps = float(last_100_df["steps"].mean())
    last_100_avg_reward = float(last_100_df["total_reward"].mean())

    if verbose:
        print("\n" + "=" * 76)
        print("                  FINAL TRAINING RUN SUMMARY STATISTICS                 ")
        print("=" * 76)
        print(f"  Total Episodes Trained     : {total_episodes}")
        print(f"  Successful Episodes        : {successful_episodes} / {total_episodes}")
        print(f"  Overall Success Rate       : {success_rate:.2f}%")
        print(f"  Last 100 Ep Success Rate   : {last_100_success_rate:.2f}%")
        print(f"  Overall Average Reward     : {avg_reward:.2f}")
        print(f"  Last 100 Ep Average Reward : {last_100_avg_reward:.2f}")
        print(f"  Overall Average Steps      : {avg_steps:.2f}")
        print(f"  Last 100 Ep Average Steps  : {last_100_avg_steps:.2f}")
        print(f"  Total Collisions           : {total_collisions} (Avg: {avg_collisions:.2f}/ep)")
        print(f"  Final Epsilon Rate         : {final_epsilon:.4f}")
        print(f"  Learned Q-Table States     : {final_non_zeros} non-zero entries (Initial: {initial_non_zeros})")
        print("=" * 76 + "\n")

    return agent, results_df


def train_multi_destinations(
    episodes_per_dest: int = 500,
    max_steps: int = config.MAX_STEPS_PER_EPISODE,
    env: Optional[CampusEnvironment] = None,
    agent: Optional[QLearningAgent] = None,
    save_model: bool = True,
    verbose: bool = True,
) -> QLearningAgent:
    """
    Train goal-conditioned policies across all predefined campus destinations (Phase 16).

    Destinations:
        0: Library (9, 9)
        1: Main Block (1, 8)
        2: Computer Lab (4, 3)
        3: Cafeteria (3, 2)
        4: Sports Ground (9, 0)

    If a pre-existing Library model exists at models/q_table.npy, Destination 0 retains
    the converged baseline weights. Destinations 1..4 are trained to optimal convergence.
    """
    if env is None:
        env = CampusEnvironment(
            rows=config.GRID_ROWS,
            cols=config.GRID_COLS,
            max_steps=max_steps,
        )
    if agent is None:
        agent = QLearningAgent(
            rows=env.rows,
            cols=env.cols,
            num_actions=config.NUM_ACTIONS,
        )
        if os.path.exists(config.Q_TABLE_SAVE_PATH):
            agent.load_model(config.Q_TABLE_SAVE_PATH)

    if verbose:
        print("=" * 76)
        print("     CAMPUS NAVIGATOR: MULTI-DESTINATION Q-LEARNING (PHASE 16)      ")
        print("=" * 76)

    start_locations = list(config.START_LOCATIONS.values())

    for dest_idx, (dest_name, dest_coord) in enumerate(config.DESTINATIONS.items()):
        agent.set_destination_id(dest_idx)
        env.set_destination(dest_coord)

        # If destination 0 is already trained and converged, retain baseline
        if dest_idx == 0 and np.count_nonzero(agent.multi_q_table[0]) >= 300:
            if verbose:
                print(f"[+] Destination {dest_idx}: '{dest_name}' {dest_coord} already converged (retained baseline).")
            continue

        if verbose:
            print(f"[>] Training Destination {dest_idx}: '{dest_name}' at {dest_coord} ({episodes_per_dest} eps)...")

        agent.epsilon = 1.0
        success_count = 0

        try:
            for ep in range(1, episodes_per_dest + 1):
                s_idx = ep % len(start_locations)
                s_pos = start_locations[s_idx]
                if s_pos == dest_coord:
                    s_pos = (0, 0)
                state = env.reset(start_pos=s_pos, goal_pos=dest_coord)

                for step in range(max_steps):
                    action = agent.choose_action(state, training=True)
                    next_state, reward, done, info = env.step(action)
                    agent.update(state, action, reward, next_state, done)
                    state = next_state
                    if done:
                        if info.get("goal_reached", False):
                            success_count += 1
                        break

                agent.decay_epsilon()
        except KeyboardInterrupt:
            if verbose:
                print(f"\n[!] Multi-destination training interrupted during '{dest_name}'.")
                print("[+] Preserving learned tensor weights to disk...")
            try:
                from utils.logger import log_warning
                log_warning(f"Multi-destination training loop interrupted by user during target '{dest_name}'. Preserving progress.")
            except ImportError:
                pass
            break

        if verbose:
            recent_sr = (success_count / episodes_per_dest) * 100.0
            print(f"    [+] '{dest_name}' trained. Success Rate: {recent_sr:.1f}% | Non-zero Q: {np.count_nonzero(agent.q_table)}")

    if save_model:
        agent.save_model(save_multi=True)
        if verbose:
            print(f"[+] Saved multi-destination model to: {config.Q_TABLE_MULTI_SAVE_PATH}")
            print(f"[+] Saved primary Library model to: {config.Q_TABLE_SAVE_PATH}")

    # Reset active destination to default (Library)
    agent.set_destination_id(0)
    env.set_destination(config.DEST_LIBRARY)
    env.set_start(config.START_MAIN_GATE)

    return agent


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for running training from terminal."""
    parser = argparse.ArgumentParser(
        description="Campus Navigator: Q-Learning Training Pipeline (Phase 3 & 16)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=config.DEFAULT_TRAINING_EPISODES,
        help="Total training episodes (default: 1000).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=config.MAX_STEPS_PER_EPISODE,
        help="Maximum steps allowed per episode (default: 200).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Interval frequency for logging progress (default: 100).",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Train policies for all 5 predefined campus destinations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.multi:
        train_multi_destinations()
    else:
        train_agent(
            episodes=args.episodes,
            max_steps=args.max_steps,
            log_interval=args.log_interval,
        )
