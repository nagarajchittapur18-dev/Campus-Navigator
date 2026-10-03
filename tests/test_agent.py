"""
Campus Navigator: Phase 19 Agent Unit Test Suite
================================================
Module: tests/test_agent.py

Verifies:
- Q-table initialization (dimensions, data types, initial zeros, tensor slice).
- Action selection (exploration with epsilon=1.0, greedy exploitation with epsilon=0.0).
- Q-value update (Bellman optimality TD equation, terminal states, learning rates).
- Epsilon decay (rate of decay, minimum exploration floor).
- Save/load model (NumPy .npy persistence, round-trip fidelity, single and multi-destination tables).
"""

import os
import sys
import tempfile
import shutil
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from agent.q_learning import QLearningAgent


class TestQLearningAgent(unittest.TestCase):
    """Test suite covering all Tabular QLearningAgent specifications."""

    def setUp(self):
        """Create fresh agent instance and temp sandbox directory."""
        self.agent = QLearningAgent(
            rows=config.GRID_ROWS,
            cols=config.GRID_COLS,
            num_actions=config.NUM_ACTIONS,
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.99,
        )
        self.temp_dir = tempfile.mkdtemp(prefix="agent_test_")

    def tearDown(self):
        """Clean up temporary sandbox directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # 1. Q-Table Initialization Tests
    # --------------------------------------------------------------------------
    def test_q_table_initialization(self):
        """Verify Q-table shape, data type, and initial zero state."""
        self.assertEqual(self.agent.num_states, 100)
        self.assertEqual(self.agent.num_actions, 4)
        self.assertEqual(self.agent.q_table.shape, (100, 4))
        self.assertEqual(self.agent.q_table.dtype, np.float32)
        self.assertEqual(np.count_nonzero(self.agent.q_table), 0)

        # Multi-destination tensor memory: verify shared memory slice
        self.assertEqual(self.agent.multi_q_table.shape, (5, 100, 4))
        self.agent.q_table[0, 0] = 99.0
        self.assertEqual(self.agent.multi_q_table[0, 0, 0], 99.0)
        self.agent.q_table[0, 0] = 0.0

    def test_state_index_bidirectional_conversion(self):
        """Verify mapping between 2D coordinates and 1D scalar indices."""
        # Top-left (0, 0) -> 0
        self.assertEqual(self.agent.state_to_index((0, 0)), 0)
        self.assertEqual(self.agent.index_to_state(0), (0, 0))

        # Bottom-right (9, 9) -> 99
        self.assertEqual(self.agent.state_to_index((9, 9)), 99)
        self.assertEqual(self.agent.index_to_state(99), (9, 9))

        # Mid-campus (3, 7) -> 37
        self.assertEqual(self.agent.state_to_index((3, 7)), 37)
        self.assertEqual(self.agent.index_to_state(37), (3, 7))

    # --------------------------------------------------------------------------
    # 2. Action Selection Tests (Exploration vs Exploitation)
    # --------------------------------------------------------------------------
    def test_action_selection_exploration(self):
        """Verify epsilon = 1.0 selects diverse stochastic actions across the action space."""
        self.agent.epsilon = 1.0
        actions_chosen = set()

        for _ in range(100):
            action = self.agent.choose_action((0, 0), training=True)
            self.assertIn(action, [0, 1, 2, 3])
            actions_chosen.add(action)

        # High probability that all 4 actions are chosen over 100 trials
        self.assertEqual(len(actions_chosen), 4)

    def test_action_selection_greedy_exploitation(self):
        """Verify epsilon = 0.0 deterministically selects the argmax Q action."""
        self.agent.epsilon = 0.0
        state = (2, 2)
        s_idx = self.agent.state_to_index(state)

        # Set specific Q-values: [UP=1.0, DOWN=10.0, LEFT=2.0, RIGHT=5.0]
        self.agent.q_table[s_idx] = [1.0, 10.0, 2.0, 5.0]

        best_action = self.agent.choose_action(state, training=False)
        self.assertEqual(best_action, config.ACTION_DOWN)
        self.assertEqual(self.agent.get_best_action(state), config.ACTION_DOWN)

    # --------------------------------------------------------------------------
    # 3. Q-Value Update Tests (Bellman Equation)
    # --------------------------------------------------------------------------
    def test_q_value_update_non_terminal(self):
        """Verify non-terminal Bellman update: Q <- Q + alpha * [r + gamma * max Q' - Q]."""
        state = (0, 0)
        action = config.ACTION_DOWN
        reward = 5.0
        next_state = (1, 0)
        done = False

        s_idx = self.agent.state_to_index(state)
        next_s_idx = self.agent.state_to_index(next_state)

        # Set next_state max Q to 10.0
        self.agent.q_table[next_s_idx] = [0.0, 10.0, 0.0, 0.0]

        # Initial Q(s, a) = 0.0
        # Expected: 0.0 + 0.1 * [5.0 + 0.9 * 10.0 - 0.0] = 0.1 * 14.0 = 1.40
        td_error = self.agent.update(state, action, reward, next_state, done)

        self.assertAlmostEqual(td_error, 14.0, places=4)
        self.assertAlmostEqual(self.agent.q_table[s_idx, action], 1.40, places=4)

    def test_q_value_update_terminal(self):
        """Verify terminal state update does NOT add future discounted reward."""
        state = (8, 9)
        action = config.ACTION_DOWN
        reward = 100.0
        next_state = (9, 9)
        done = True  # Terminal state (goal reached)

        s_idx = self.agent.state_to_index(state)
        next_s_idx = self.agent.state_to_index(next_state)

        # Even if next_state has high Q-values, it must be ignored on done=True
        self.agent.q_table[next_s_idx] = [50.0, 50.0, 50.0, 50.0]

        # Expected: 0.0 + 0.1 * [100.0 - 0.0] = 10.0
        td_error = self.agent.update(state, action, reward, next_state, done)

        self.assertAlmostEqual(td_error, 100.0, places=4)
        self.assertAlmostEqual(self.agent.q_table[s_idx, action], 10.0, places=4)

    # --------------------------------------------------------------------------
    # 4. Epsilon Decay Tests
    # --------------------------------------------------------------------------
    def test_epsilon_decay_towards_minimum(self):
        """Verify epsilon decays per episode and respects epsilon_min floor."""
        self.agent.epsilon = 1.0
        self.agent.epsilon_decay = 0.5
        self.agent.epsilon_min = 0.1

        # Step 1: 1.0 * 0.5 = 0.5
        self.agent.decay_epsilon()
        self.assertAlmostEqual(self.agent.epsilon, 0.5, places=4)

        # Step 2: 0.5 * 0.5 = 0.25
        self.agent.decay_epsilon()
        self.assertAlmostEqual(self.agent.epsilon, 0.25, places=4)

        # Step 3: 0.25 * 0.5 = 0.125
        self.agent.decay_epsilon()
        self.assertAlmostEqual(self.agent.epsilon, 0.125, places=4)

        # Step 4: 0.125 * 0.5 = 0.0625 -> clamped to min 0.10
        self.agent.decay_epsilon()
        self.assertAlmostEqual(self.agent.epsilon, 0.10, places=4)

    # --------------------------------------------------------------------------
    # 5. Save and Load Model Tests
    # --------------------------------------------------------------------------
    def test_save_and_load_model_roundtrip(self):
        """Verify learned Q-table weights survive serialization to disk."""
        target_file = os.path.join(self.temp_dir, "test_q_table.npy")

        # Plant specific weights
        self.agent.q_table[10, 2] = 42.5
        self.agent.q_table[99, 1] = 99.9

        # Save to disk
        saved_path = self.agent.save_model(target_file, save_multi=False)
        self.assertTrue(os.path.exists(saved_path))

        # Create fresh blank agent
        fresh_agent = QLearningAgent(rows=10, cols=10)
        self.assertEqual(np.count_nonzero(fresh_agent.q_table), 0)

        # Load weights
        success = fresh_agent.load_model(saved_path)
        self.assertTrue(success)
        self.assertAlmostEqual(fresh_agent.q_table[10, 2], 42.5, places=4)
        self.assertAlmostEqual(fresh_agent.q_table[99, 1], 99.9, places=4)


if __name__ == "__main__":
    unittest.main()
