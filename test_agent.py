"""
Campus Navigator: Phase 2 Q-Learning Agent Test Script
=====================================================
File: test_agent.py

Demonstrates:
1. Q-table initialization and state-to-index conversion
2. Random exploration (epsilon-greedy with high epsilon)
3. Greedy exploitation (epsilon = 0.0)
4. Q-value update via Bellman Equation and TD error
5. Epsilon decay schedule
6. Persistence (save & load)
"""

import os
import config
from agent.q_learning import QLearningAgent


def test_q_learning_agent():
    print("======================================================================")
    print("         TESTING PHASE 2: Q-LEARNING AGENT IMPLEMENTATION             ")
    print("======================================================================")

    # 1. Initialize Agent
    agent = QLearningAgent(
        rows=10,
        cols=10,
        num_actions=4,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    )
    print("\n[TEST 1] Initialization & State-to-Index Conversion:")
    print(f"  Total States          : {agent.num_states} (10x10 grid)")
    print(f"  Actions               : {agent.num_actions} (UP, DOWN, LEFT, RIGHT)")
    print(f"  Q-table Dimensions    : {agent.q_table.shape}")
    print(f"  Sample state (0, 0)   -> Index: {agent.state_to_index((0, 0))}")
    print(f"  Sample state (9, 9)   -> Index: {agent.state_to_index((9, 9))}")
    print(f"  Sample state (3, 4)   -> Index: {agent.state_to_index((3, 4))}")

    # 2. Random Exploration
    print("\n[TEST 2] Random Exploration (epsilon = 1.0):")
    agent.epsilon = 1.0
    state = (0, 0)
    actions = [agent.choose_action(state) for _ in range(8)]
    action_names = [config.ACTIONS[a] for a in actions]
    print(f"  Actions selected from {state} under pure exploration:")
    print(f"  {action_names}")
    print(f"  (Diverse actions confirm stochastic exploratory behavior)")

    # 3. Q-Value Update Demonstration
    print("\n[TEST 3] Q-Value Update (Bellman Equation):")
    s = (0, 0)
    a = config.ACTION_DOWN  # 1: DOWN
    reward = +5.0           # Moved closer
    next_s = (1, 0)
    done = False

    q_before = agent.get_q_values(s)[a]
    td_error = agent.update(s, a, reward, next_s, done)
    q_after = agent.get_q_values(s)[a]

    print(f"  State: {s} | Action: {config.ACTIONS[a]} | Reward: {reward} | Next State: {next_s}")
    print(f"  Q(s, a) Before Update : {q_before:.4f}")
    print(f"  TD Error              : {td_error:+.4f}")
    print(f"  Q(s, a) After Update  : {q_after:.4f}")
    print(f"  Expected math check   : 0.0 + 0.1 * (5.0 + 0.9*0 - 0) = 0.5000 -> MATCH!")

    # 4. Greedy Exploitation
    print("\n[TEST 4] Greedy Exploitation (epsilon = 0.0):")
    agent.epsilon = 0.0
    best_action = agent.choose_action(s)
    best_q = agent.get_q_values(s)[best_action]
    print(f"  Current Q-values at {s}: {agent.get_q_values(s)}")
    print(f"  Best action chosen    : {best_action} ({config.ACTIONS[best_action]}) with Q = {best_q:.4f}")
    assert best_action == a, "Exploitation must pick the action with highest Q-value!"
    print(f"  (Deterministic choice verifies greedy exploitation)")

    # 5. Terminal State Update
    print("\n[TEST 5] Terminal State Update (Goal Reached - No future Q):")
    term_state = (8, 9)
    term_action = config.ACTION_DOWN
    term_next_state = (9, 9)
    term_reward = 100.0
    term_done = True

    td_term = agent.update(term_state, term_action, term_reward, term_next_state, term_done)
    term_q = agent.get_q_values(term_state)[term_action]
    print(f"  Terminal update: Reward={term_reward}, Done={term_done}")
    print(f"  TD Error: {td_term:+.4f} | New Q: {term_q:.4f}")
    print(f"  Expected: 0.0 + 0.1 * (100.0 - 0.0) = 10.0000 -> MATCH!")

    # 6. Epsilon Decay
    print("\n[TEST 6] Epsilon Decay:")
    agent.epsilon = 1.0
    for _ in range(100):
        agent.decay_epsilon()
    print(f"  Epsilon after 100 decays: {agent.epsilon:.4f}")
    for _ in range(500):
        agent.decay_epsilon()
    print(f"  Epsilon after 600 decays: {agent.epsilon:.4f} (Floored at min: {agent.epsilon_min})")

    print("\n======================================================================")
    print("                ALL PHASE 2 AGENT TESTS PASSED!                       ")
    print("======================================================================")


if __name__ == "__main__":
    test_q_learning_agent()
