"""
Campus Navigator: Phase 19 Evaluation Unit Test Suite
=====================================================
Module: tests/test_evaluation.py

Verifies:
- Trained model loads (evaluation pipeline loads weights from .npy file).
- Evaluation does not modify Q-table (Q-table weights are strictly frozen during evaluation).
- Metrics are calculated (success rate, average reward, average steps, collisions, summary files).
"""

import os
import sys
import json
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
from training.evaluate import evaluate_agent


class TestEvaluationPipeline(unittest.TestCase):
    """Test suite covering RL policy evaluation guarantees and metrics."""

    def setUp(self):
        """Set up test environment, agent, and sandbox directory."""
        self.temp_dir = tempfile.mkdtemp(prefix="eval_test_")
        self.model_path = os.path.join(self.temp_dir, "eval_q_table.npy")

        # Create synthetic deterministic trained policy from (0,0) down to (9,9)
        trained_q = np.zeros((100, 4), dtype=np.float32)
        # Give positive Q to DOWN (action 1) and RIGHT (action 3)
        for r in range(10):
            for c in range(10):
                s_idx = r * 10 + c
                trained_q[s_idx, config.ACTION_DOWN] = 50.0
                trained_q[s_idx, config.ACTION_RIGHT] = 50.0
        np.save(self.model_path, trained_q)

    def tearDown(self):
        """Clean up temporary sandbox directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # 1. Trained Model Loads
    # --------------------------------------------------------------------------
    def test_trained_model_loads_successfully(self):
        """Verify evaluate_agent successfully loads Q-table from the specified path."""
        agent = QLearningAgent(rows=10, cols=10)
        self.assertEqual(np.count_nonzero(agent.q_table), 0)

        # Run minimal evaluation with custom model
        summary = evaluate_agent(
            episodes=2,
            max_steps=20,
            model_path=self.model_path,
            agent=agent,
            save_csv=False,
            save_json=False,
            verbose=False,
        )

        self.assertIsNotNone(summary)
        # Q-table was loaded from disk into agent
        self.assertGreater(np.count_nonzero(agent.q_table), 0)
        self.assertEqual(agent.epsilon, 0.0)

    # --------------------------------------------------------------------------
    # 2. Evaluation Does NOT Modify Q-Table (Strictly Frozen Weights)
    # --------------------------------------------------------------------------
    def test_evaluation_does_not_modify_q_table(self):
        """Verify RL requirement that Q-table weights remain completely unchanged during evaluation."""
        agent = QLearningAgent(rows=10, cols=10)
        agent.load_model(self.model_path)

        # Snapshot Q-table state before evaluation
        q_snapshot_before = agent.q_table.copy()
        multi_snapshot_before = agent.multi_q_table.copy()

        # Run 25 evaluation episodes
        evaluate_agent(
            episodes=25,
            max_steps=30,
            agent=agent,
            save_csv=False,
            save_json=False,
            verbose=False,
        )

        # Q-table weights must be 100% byte-for-byte identical
        self.assertTrue(
            np.array_equal(agent.q_table, q_snapshot_before),
            "CRITICAL FAULT: Q-table weights were modified during policy evaluation!"
        )
        self.assertTrue(
            np.array_equal(agent.multi_q_table, multi_snapshot_before),
            "CRITICAL FAULT: Multi-destination tensor was modified during evaluation!"
        )
        # Exploration rate must remain frozen at 0.0
        self.assertEqual(agent.epsilon, 0.0)

    # --------------------------------------------------------------------------
    # 3. Metrics Are Calculated
    # --------------------------------------------------------------------------
    def test_metrics_are_calculated_and_persisted(self):
        """Verify summary metrics and evaluation data artifacts are generated accurately."""
        eval_csv = os.path.join(self.temp_dir, "eval_results.csv")
        eval_json = os.path.join(self.temp_dir, "eval_summary.json")

        with patch("config.EVAL_CSV_PATH", eval_csv), \
             patch("config.EVAL_SUMMARY_PATH", eval_json), \
             patch("config.EVAL_RESULTS_PATH", eval_json):

            summary = evaluate_agent(
                episodes=5,
                max_steps=20,
                model_path=self.model_path,
                save_csv=True,
                save_json=True,
                verbose=False,
            )

        # Verify summary dictionary structure
        required_keys = [
            "episodes",
            "success_rate",
            "avg_reward",
            "avg_steps",
            "min_steps",
            "max_steps",
            "collision_rate",
            "learned_path_length",
            "optimal_path",
        ]
        for key in required_keys:
            self.assertIn(key, summary)

        self.assertEqual(summary["episodes"], 5)
        self.assertIsInstance(summary["success_rate"], float)
        self.assertIsInstance(summary["avg_reward"], float)
        self.assertIsInstance(summary["avg_steps"], float)
        self.assertIsInstance(summary["optimal_path"], list)

        # Verify persisted CSV
        self.assertTrue(os.path.exists(eval_csv))
        df = pd.read_csv(eval_csv)
        self.assertEqual(len(df), 5)
        for col in ["episode", "steps", "total_reward", "success", "collisions"]:
            self.assertIn(col, df.columns)

        # Verify persisted JSON summary
        self.assertTrue(os.path.exists(eval_json))
        with open(eval_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["episodes"], 5)


if __name__ == "__main__":
    unittest.main()
