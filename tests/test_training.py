"""
Campus Navigator: Phase 19 Training Unit Test Suite
===================================================
Module: tests/test_training.py

Verifies:
- Training starts (initializes environment, agent state, step recording).
- Training finishes (runs requested episode horizon, decays epsilon, completes gracefully).
- Results are generated (saves Q-table .npy and training_results.csv with required columns).
"""

import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from environment.campus_env import CampusEnvironment
from agent.q_learning import QLearningAgent
from training.train import train_agent, train_multi_destinations


class TestTrainingPipeline(unittest.TestCase):
    """Test suite covering tabular Q-learning training lifecycle."""

    def setUp(self):
        """Create fresh environment, agent, and temporary sandbox directory."""
        self.env = CampusEnvironment(
            rows=config.GRID_ROWS,
            cols=config.GRID_COLS,
            max_steps=50,
        )
        self.agent = QLearningAgent(
            rows=config.GRID_ROWS,
            cols=config.GRID_COLS,
            num_actions=config.NUM_ACTIONS,
            learning_rate=0.1,
            discount_factor=0.9,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.95,
        )
        self.temp_dir = tempfile.mkdtemp(prefix="training_test_")

    def tearDown(self):
        """Clean up temporary sandbox directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # 1. Training Starts
    # --------------------------------------------------------------------------
    def test_training_starts(self):
        """Verify training loop starts at episode 1 and initiates experiential updates."""
        # Initial agent Q-table has 0 non-zero entries
        self.assertEqual(np.count_nonzero(self.agent.q_table), 0)

        # Run minimal 1-episode training
        temp_model = os.path.join(self.temp_dir, "test_q.npy")
        temp_csv = os.path.join(self.temp_dir, "test_res.csv")

        with patch("config.Q_TABLE_SAVE_PATH", temp_model), \
             patch("config.TRAINING_RESULTS_PATH", temp_csv), \
             patch("config.TRAINING_RESULTS_SAVE_PATH", temp_csv):

            trained_agent, df = train_agent(
                env=self.env,
                agent=self.agent,
                episodes=1,
                max_steps=10,
                verbose=False,
                save_model=True,
                save_results=True,
            )

        self.assertIsNotNone(trained_agent)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["episode"], 1)
        self.assertGreater(df.iloc[0]["steps"], 0)
        # Agent should have updated at least one state-action entry
        self.assertGreater(np.count_nonzero(trained_agent.q_table), 0)

    # --------------------------------------------------------------------------
    # 2. Training Finishes
    # --------------------------------------------------------------------------
    def test_training_finishes_full_horizon(self):
        """Verify training runs all specified episodes and decays exploration rate."""
        episodes_to_run = 8
        initial_eps = self.agent.epsilon

        temp_model = os.path.join(self.temp_dir, "test_q_fin.npy")
        temp_csv = os.path.join(self.temp_dir, "test_res_fin.csv")

        with patch("config.Q_TABLE_SAVE_PATH", temp_model), \
             patch("config.TRAINING_RESULTS_PATH", temp_csv), \
             patch("config.TRAINING_RESULTS_SAVE_PATH", temp_csv):

            trained_agent, df = train_agent(
                env=self.env,
                agent=self.agent,
                episodes=episodes_to_run,
                max_steps=20,
                verbose=False,
                save_model=True,
                save_results=True,
            )

        self.assertEqual(len(df), episodes_to_run)
        self.assertEqual(int(df.iloc[-1]["episode"]), episodes_to_run)
        # Epsilon decayed from initial 1.0
        self.assertLess(trained_agent.epsilon, initial_eps)

    # --------------------------------------------------------------------------
    # 3. Results Are Generated
    # --------------------------------------------------------------------------
    def test_results_are_generated_and_saved(self):
        """Verify metrics CSV and model weights are persisted to disk with correct schema."""
        model_path = os.path.join(self.temp_dir, "saved_q_table.npy")
        results_csv = os.path.join(self.temp_dir, "saved_training_results.csv")

        with patch("config.Q_TABLE_SAVE_PATH", model_path), \
             patch("config.TRAINING_RESULTS_PATH", results_csv), \
             patch("config.TRAINING_RESULTS_SAVE_PATH", results_csv):

            train_agent(
                env=self.env,
                agent=self.agent,
                episodes=5,
                max_steps=15,
                verbose=False,
                save_model=True,
                save_results=True,
            )

        # 1. Model binary file verification
        self.assertTrue(os.path.exists(model_path))
        loaded_q = np.load(model_path)
        self.assertEqual(loaded_q.shape, (100, 4))
        self.assertEqual(loaded_q.dtype, np.float32)

        # 2. Results CSV verification
        self.assertTrue(os.path.exists(results_csv))
        loaded_df = pd.read_csv(results_csv)

        expected_columns = ["episode", "steps", "total_reward", "success", "collisions", "epsilon"]
        for col in expected_columns:
            self.assertIn(col, loaded_df.columns)

        self.assertEqual(len(loaded_df), 5)


if __name__ == "__main__":
    unittest.main()
