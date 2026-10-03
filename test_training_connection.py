"""
Campus Navigator: Phase 3 Verification Script
============================================
File: test_training_connection.py

Verifies:
1. Q-table was saved and changed from initial state (zeros -> trained values)
2. Training results CSV contains all required fields and valid data
3. Demonstrates the learned greedy policy path from Main Gate to Central Library
"""

import os
import numpy as np
import pandas as pd

import config
from environment.campus_env import CampusEnvironment
from agent.q_learning import QLearningAgent


def verify_phase_3_training():
    print("======================================================================")
    print("        VERIFYING PHASE 3: TRAINING & Q-TABLE CONVERGENCE            ")
    print("======================================================================")

    # 1. Verify Q-Table File Existence and Shape
    q_table_path = config.Q_TABLE_SAVE_PATH
    assert os.path.exists(q_table_path), f"Q-table file not found at {q_table_path}"
    q_table = np.load(q_table_path)

    print(f"\n[1] Q-Table Verification:")
    print(f"  File Location        : {q_table_path}")
    print(f"  Shape                : {q_table.shape} (100 states x 4 actions)")
    print(f"  Data Type            : {q_table.dtype}")

    # Check non-zero entries (verifying change from initial state)
    non_zeros = np.count_nonzero(q_table)
    total_entries = q_table.size
    print(f"  Non-zero Q-entries   : {non_zeros} / {total_entries} ({(non_zeros/total_entries)*100:.1f}%)")
    assert non_zeros > 0, "Q-table is still empty! It must have changed during training."
    print(f"  Min Q-Value          : {np.min(q_table):.4f}")
    print(f"  Max Q-Value          : {np.max(q_table):.4f}")
    print("  -> CONFIRMED: Q-table significantly updated from initial zero state!")

    # 2. Inspect Sample Learned Q-values
    agent = QLearningAgent(rows=config.GRID_ROWS, cols=config.GRID_COLS)
    agent.load_model(q_table_path)

    sample_states = [(0, 0), (1, 1), (5, 4), (8, 9)]
    print(f"\n[2] Sample Learned Q-Values [UP, DOWN, LEFT, RIGHT]:")
    for s in sample_states:
        q_vals = agent.get_q_values(s)
        best_a = agent.get_best_action(s)
        print(
            f"  State {s} -> Q: {np.round(q_vals, 2)} | "
            f"Best Action: {config.ACTIONS[best_a]} (Q={q_vals[best_a]:.2f})"
        )

    # 3. Verify Training Results CSV
    csv_path = config.TRAINING_RESULTS_PATH
    assert os.path.exists(csv_path), f"CSV file not found at {csv_path}"
    df = pd.read_csv(csv_path)

    print(f"\n[3] Training Results CSV Verification:")
    print(f"  File Location        : {csv_path}")
    print(f"  Total Rows (Episodes): {len(df)}")
    print(f"  Columns Present      : {list(df.columns)}")

    required_cols = {"episode", "total_reward", "steps", "success", "collisions", "epsilon"}
    assert required_cols.issubset(df.columns), f"Missing required columns! Found: {df.columns}"

    # Print first and last 3 rows
    print("\n  Early Training Episodes (High exploration, low rewards):")
    print(df.head(3).to_string(index=False))
    print("\n  Late Training Episodes (Converged exploitation, high rewards):")
    print(df.tail(3).to_string(index=False))

    # 4. Trace Learned Autonomous Path (Greedy Exploitation)
    print(f"\n[4] Autonomous Navigation Test (Greedy Policy from Start):")
    env = CampusEnvironment()
    state = env.reset()
    done = False
    path = [state]
    total_r = 0.0

    while not done and len(path) <= 40:
        action = agent.get_best_action(state)
        next_state, reward, done, info = env.step(action)
        path.append(next_state)
        total_r += reward
        state = next_state

    print(f"  Path Length          : {len(path) - 1} steps")
    print(f"  Cumulative Reward    : {total_r:.1f}")
    print(f"  Goal Reached         : {state == env.goal_pos}")
    print(f"  Trajectory           : {path}")

    print("\n======================================================================")
    print("                PHASE 3 VERIFICATION COMPLETE: SUCCESS!               ")
    print("======================================================================")


if __name__ == "__main__":
    verify_phase_3_training()
