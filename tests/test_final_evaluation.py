"""
Campus Navigator: Phase 22 Final Evaluation Test Suite
======================================================
Module: tests/test_final_evaluation.py

Verifies:
- extract_actual_training_episodes accurately parses training history.
- evaluate_trained_model executes zero-exploration policy evaluation with frozen weights.
- generate_final_evaluation_report generates:
  * results/final_report.json with all required project/algorithm/metric fields.
  * results/final_results.csv with all required column schemas.
- Correct reporting of:
  * PROJECT: Campus Navigator
  * ALGORITHM: Q-Learning
  * ENVIRONMENT: 10x10 campus grid
  * TRAINING EPISODES: actual number
  * FINAL SUCCESS RATE: actual value
  * AVERAGE REWARD: actual value
  * AVERAGE STEPS: actual value
  * COLLISION RATE: actual value
"""

import json
import os
import sys
import tempfile
import shutil
import unittest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from training.final_evaluation import (
    evaluate_trained_model,
    extract_actual_training_episodes,
    generate_final_evaluation_report,
)


class TestFinalEvaluation(unittest.TestCase):
    """Test suite covering Phase 22 automated final evaluation and reporting."""

    def setUp(self):
        """Create sandbox directory for test artifacts."""
        self.temp_dir = tempfile.mkdtemp(prefix="final_eval_test_")
        self.json_path = os.path.join(self.temp_dir, "final_report.json")
        self.csv_path = os.path.join(self.temp_dir, "final_results.csv")
        self.model_path = os.path.join(self.temp_dir, "test_q_table.npy")
        self.train_csv_path = os.path.join(self.temp_dir, "test_train.csv")

        # Create synthetic trained policy weights directed toward goal
        trained_q = np.zeros((100, 4), dtype=np.float32)
        for r in range(10):
            for c in range(10):
                s_idx = r * 10 + c
                trained_q[s_idx, config.ACTION_DOWN] = 50.0
                trained_q[s_idx, config.ACTION_RIGHT] = 50.0
        np.save(self.model_path, trained_q)

        # Create synthetic training results CSV (600 episodes)
        train_rows = [
            {"episode": i, "steps": 25, "total_reward": 100.0, "success": 1, "collisions": 2, "epsilon": 0.05}
            for i in range(1, 601)
        ]
        pd.DataFrame(train_rows).to_csv(self.train_csv_path, index=False)

    def tearDown(self):
        """Clean up sandbox directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # 1. Training History Parsing
    # --------------------------------------------------------------------------
    def test_extract_actual_training_episodes(self):
        """Verify extraction of actual episode count and training stats from CSV."""
        episodes, summary = extract_actual_training_episodes(self.train_csv_path)
        self.assertEqual(episodes, 600)
        self.assertTrue(summary["file_exists"])
        self.assertEqual(summary["total_episodes"], 600)
        self.assertEqual(summary["training_success_rate"], 100.0)
        self.assertEqual(summary["training_avg_steps"], 25.0)

        # Test fallback when file does not exist
        missing_csv = os.path.join(self.temp_dir, "nonexistent.csv")
        fallback_eps, fallback_summary = extract_actual_training_episodes(missing_csv, default_episodes=500)
        self.assertEqual(fallback_eps, 500)
        self.assertFalse(fallback_summary["file_exists"])

    # --------------------------------------------------------------------------
    # 2. Frozen Policy Evaluation
    # --------------------------------------------------------------------------
    def test_evaluate_trained_model(self):
        """Verify evaluate_trained_model computes all required evaluation statistics."""
        metrics = evaluate_trained_model(
            model_path=self.model_path,
            eval_episodes=10,
            max_steps=50,
        )

        self.assertIn("success_rate", metrics)
        self.assertIn("average_reward", metrics)
        self.assertIn("average_steps", metrics)
        self.assertIn("collision_rate", metrics)
        self.assertIn("optimal_route_steps", metrics)
        self.assertIn("optimal_route_path", metrics)
        self.assertIn("episode_records", metrics)

        self.assertIsInstance(metrics["success_rate"], float)
        self.assertIsInstance(metrics["average_reward"], float)
        self.assertIsInstance(metrics["average_steps"], float)
        self.assertIsInstance(metrics["collision_rate"], float)
        self.assertEqual(len(metrics["episode_records"]), 10)

    # --------------------------------------------------------------------------
    # 3. Final Report JSON Generation & Schema
    # --------------------------------------------------------------------------
    def test_generate_final_evaluation_report_json(self):
        """Verify final_report.json contains all required Phase 22 fields."""
        report, _ = generate_final_evaluation_report(
            model_path=self.model_path,
            training_csv=self.train_csv_path,
            eval_episodes=20,
            save_json=True,
            save_csv=True,
            json_path=self.json_path,
            csv_path=self.csv_path,
            verbose=False,
        )

        self.assertTrue(os.path.exists(self.json_path))

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Verify required specifications:
        # PROJECT: Campus Navigator
        self.assertEqual(data["PROJECT"], "Campus Navigator")
        self.assertEqual(data["project"], "Campus Navigator")

        # ALGORITHM: Q-Learning
        self.assertEqual(data["ALGORITHM"], "Q-Learning")
        self.assertEqual(data["algorithm"], "Q-Learning")

        # ENVIRONMENT: 10x10 campus grid
        self.assertEqual(data["ENVIRONMENT"], "10x10 campus grid")
        self.assertEqual(data["environment"], "10x10 campus grid")

        # TRAINING EPISODES: actual number
        self.assertEqual(data["TRAINING EPISODES"], 600)
        self.assertEqual(data["training_episodes"], 600)

        # FINAL SUCCESS RATE: actual value
        self.assertIn("FINAL SUCCESS RATE", data)
        self.assertIn("final_success_rate", data)
        self.assertIsInstance(data["final_success_rate"], (int, float))

        # AVERAGE REWARD: actual value
        self.assertIn("AVERAGE REWARD", data)
        self.assertIn("average_reward", data)
        self.assertIsInstance(data["average_reward"], (int, float))

        # AVERAGE STEPS: actual value
        self.assertIn("AVERAGE STEPS", data)
        self.assertIn("average_steps", data)
        self.assertIsInstance(data["average_steps"], (int, float))

        # COLLISION RATE: actual value
        self.assertIn("COLLISION RATE", data)
        self.assertIn("collision_rate", data)
        self.assertIsInstance(data["collision_rate"], (int, float))

        # Telemetry & Route
        self.assertIn("optimal_route_steps", data)
        self.assertIn("optimal_route_path", data)
        self.assertIn("training_history_summary", data)

    # --------------------------------------------------------------------------
    # 4. Final Results CSV Generation & Schema
    # --------------------------------------------------------------------------
    def test_generate_final_results_csv(self):
        """Verify final_results.csv contains all required columns and valid records."""
        generate_final_evaluation_report(
            model_path=self.model_path,
            training_csv=self.train_csv_path,
            eval_episodes=20,
            save_json=True,
            save_csv=True,
            json_path=self.json_path,
            csv_path=self.csv_path,
            verbose=False,
        )

        self.assertTrue(os.path.exists(self.csv_path))
        df = pd.read_csv(self.csv_path)
        self.assertGreaterEqual(len(df), 1)

        required_columns = [
            "project",
            "algorithm",
            "environment",
            "training_episodes",
            "final_success_rate",
            "average_reward",
            "average_steps",
            "collision_rate",
        ]
        for col in required_columns:
            self.assertIn(col, df.columns)

        # Row 1 values
        row = df.iloc[0]
        self.assertEqual(row["project"], "Campus Navigator")
        self.assertEqual(row["algorithm"], "Q-Learning")
        self.assertEqual(row["environment"], "10x10 campus grid")
        self.assertEqual(row["training_episodes"], 600)
        self.assertIsInstance(float(row["final_success_rate"]), float)
        self.assertIsInstance(float(row["average_reward"]), float)
        self.assertIsInstance(float(row["average_steps"]), float)
        self.assertIsInstance(float(row["collision_rate"]), float)

        # Detailed episodes CSV also created
        episodes_csv = os.path.join(self.temp_dir, "final_evaluation_episodes.csv")
        self.assertTrue(os.path.exists(episodes_csv))
        ep_df = pd.read_csv(episodes_csv)
        self.assertEqual(len(ep_df), 20)


if __name__ == "__main__":
    unittest.main()
