"""
Campus Navigator: Phase 20 RL Experiments Test Suite
=====================================================
Module: tests/test_experiments.py

Verifies:
- ExperimentConfig initialization, defaults, and custom overrides.
- Single experiment execution (run_single_experiment) trains and evaluates policy.
- Accurate metric calculations:
  * Success rate
  * Average reward
  * Average steps
  * Collision rate
- Results CSV generation and schema compliance (results/experiments.csv).
- Comparison plots generation (bar chart dashboard and learning curve overlays).
- Factual neutrality (measured metrics only, no arbitrary winner assertion).
"""

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
from experiments.framework import (
    ExperimentConfig,
    ExperimentResult,
    run_single_experiment,
    plot_experiment_comparison,
    plot_learning_curves_comparison,
    format_experiment_summary_table,
)


class TestRLExperiments(unittest.TestCase):
    """Test suite covering Phase 20 experiment framework, execution, and outputs."""

    def setUp(self):
        """Create sandbox directory for test artifacts."""
        self.temp_dir = tempfile.mkdtemp(prefix="exp_test_")
        self.csv_path = os.path.join(self.temp_dir, "test_experiments.csv")
        self.plot_bar_path = os.path.join(self.temp_dir, "test_comparison.png")
        self.plot_curve_path = os.path.join(self.temp_dir, "test_learning_curves.png")

    def tearDown(self):
        """Clean up sandbox directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # 1. ExperimentConfig Verification
    # --------------------------------------------------------------------------
    def test_experiment_config_defaults_and_overrides(self):
        """Verify ExperimentConfig sets proper hyperparameters and supports custom values."""
        cfg_default = ExperimentConfig(name="Default Exp", episodes=500)
        self.assertEqual(cfg_default.name, "Default Exp")
        self.assertEqual(cfg_default.episodes, 500)
        self.assertEqual(cfg_default.learning_rate, config.LEARNING_RATE)
        self.assertEqual(cfg_default.discount_factor, config.DISCOUNT_FACTOR)
        self.assertEqual(cfg_default.eval_episodes, 100)

        cfg_custom = ExperimentConfig(
            name="Custom 1000",
            episodes=1000,
            learning_rate=0.2,
            discount_factor=0.95,
            eval_episodes=50,
        )
        self.assertEqual(cfg_custom.episodes, 1000)
        self.assertEqual(cfg_custom.learning_rate, 0.2)
        self.assertEqual(cfg_custom.discount_factor, 0.95)
        self.assertEqual(cfg_custom.eval_episodes, 50)

    # --------------------------------------------------------------------------
    # 2. Single Experiment Execution & Metric Calculations
    # --------------------------------------------------------------------------
    def test_run_single_experiment_computes_all_required_metrics(self):
        """Verify run_single_experiment produces complete empirical metrics."""
        model_sandbox = os.path.join(self.temp_dir, "mini_model.npy")
        cfg = ExperimentConfig(
            name="Mini Test Exp",
            episodes=30,
            max_steps=50,
            eval_episodes=10,
            save_model=True,
            model_save_path=model_sandbox,
        )

        result = run_single_experiment(cfg, verbose=False)

        self.assertIsInstance(result, ExperimentResult)
        self.assertEqual(result.experiment_name, "Mini Test Exp")
        self.assertEqual(result.episodes, 30)

        # Check required Phase 20 metrics:
        # 1. Success Rate
        self.assertIsInstance(result.training_success_rate, float)
        self.assertIsInstance(result.eval_success_rate, float)
        self.assertGreaterEqual(result.training_success_rate, 0.0)
        self.assertLessEqual(result.training_success_rate, 100.0)
        self.assertGreaterEqual(result.eval_success_rate, 0.0)
        self.assertLessEqual(result.eval_success_rate, 100.0)

        # 2. Average Reward
        self.assertIsInstance(result.training_avg_reward, float)
        self.assertIsInstance(result.eval_avg_reward, float)

        # 3. Average Steps
        self.assertIsInstance(result.training_avg_steps, float)
        self.assertIsInstance(result.eval_avg_steps, float)
        self.assertGreater(result.training_avg_steps, 0.0)
        self.assertGreater(result.eval_avg_steps, 0.0)

        # 4. Collision Rate
        self.assertIsInstance(result.training_collision_rate, float)
        self.assertIsInstance(result.eval_collision_rate, float)
        self.assertGreaterEqual(result.training_collision_rate, 0.0)
        self.assertGreaterEqual(result.eval_collision_rate, 0.0)

        # Model artifact was created
        self.assertTrue(os.path.exists(model_sandbox))

    # --------------------------------------------------------------------------
    # 3. CSV Persistence and Schema Compliance
    # --------------------------------------------------------------------------
    def test_results_csv_schema_and_persistence(self):
        """Verify experiment metrics convert to a valid DataFrame and persist to CSV."""
        r1 = ExperimentResult(
            experiment_name="Exp 500",
            episodes=500,
            max_steps=200,
            learning_rate=0.1,
            discount_factor=0.9,
            final_epsilon=0.08,
            training_success_rate=94.2,
            training_avg_reward=98.7,
            training_avg_steps=45.5,
            training_collision_rate=8.1,
            total_training_collisions=4062,
            successful_training_episodes=471,
            recent_100_success_rate=100.0,
            recent_100_avg_reward=166.7,
            recent_100_avg_steps=20.8,
            recent_100_collision_rate=0.6,
            eval_episodes=100,
            eval_success_rate=100.0,
            eval_avg_reward=172.0,
            eval_avg_steps=18.0,
            eval_collision_rate=0.0,
            eval_min_steps=18,
            eval_max_steps=18,
            eval_optimal_route_length=19,
            best_episode=203,
            best_reward=184.0,
            non_zero_q_entries=320,
            execution_time_seconds=0.45,
            training_records=[],
        )

        r2 = ExperimentResult(
            experiment_name="Exp 1000",
            episodes=1000,
            max_steps=200,
            learning_rate=0.1,
            discount_factor=0.9,
            final_epsilon=0.007,
            training_success_rate=97.7,
            training_avg_reward=131.9,
            training_avg_steps=31.6,
            training_collision_rate=4.3,
            total_training_collisions=4327,
            successful_training_episodes=977,
            recent_100_success_rate=100.0,
            recent_100_avg_reward=169.8,
            recent_100_avg_steps=19.4,
            recent_100_collision_rate=0.3,
            eval_episodes=100,
            eval_success_rate=100.0,
            eval_avg_reward=172.0,
            eval_avg_steps=18.0,
            eval_collision_rate=0.0,
            eval_min_steps=18,
            eval_max_steps=18,
            eval_optimal_route_length=19,
            best_episode=416,
            best_reward=190.0,
            non_zero_q_entries=317,
            execution_time_seconds=0.70,
            training_records=[],
        )

        df = pd.DataFrame([r1.to_dict(), r2.to_dict()])
        df.to_csv(self.csv_path, index=False)

        self.assertTrue(os.path.exists(self.csv_path))
        loaded_df = pd.read_csv(self.csv_path)
        self.assertEqual(len(loaded_df), 2)

        # Required columns must be present
        required_cols = [
            "experiment_name",
            "episodes",
            "training_success_rate",
            "training_avg_reward",
            "training_avg_steps",
            "training_collision_rate",
            "eval_success_rate",
            "eval_avg_reward",
            "eval_avg_steps",
            "eval_collision_rate",
        ]
        for col in required_cols:
            self.assertIn(col, loaded_df.columns)

    # --------------------------------------------------------------------------
    # 4. Comparison Plots Generation
    # --------------------------------------------------------------------------
    def test_plot_generation_creates_valid_image_files(self):
        """Verify comparison plots are generated cleanly to disk."""
        # Create minimal synthetic history
        history_500 = [{"episode": i, "reward": 50 + (i * 0.2)} for i in range(50)]
        history_1000 = [{"episode": i, "reward": 50 + (i * 0.25)} for i in range(50)]

        r1 = ExperimentResult(
            experiment_name="500 eps",
            episodes=500,
            max_steps=200,
            learning_rate=0.1,
            discount_factor=0.9,
            final_epsilon=0.08,
            training_success_rate=94.0,
            training_avg_reward=100.0,
            training_avg_steps=40.0,
            training_collision_rate=5.0,
            total_training_collisions=200,
            successful_training_episodes=470,
            recent_100_success_rate=100.0,
            recent_100_avg_reward=160.0,
            recent_100_avg_steps=20.0,
            recent_100_collision_rate=0.5,
            eval_episodes=50,
            eval_success_rate=100.0,
            eval_avg_reward=172.0,
            eval_avg_steps=18.0,
            eval_collision_rate=0.0,
            eval_min_steps=18,
            eval_max_steps=18,
            eval_optimal_route_length=19,
            best_episode=200,
            best_reward=180.0,
            non_zero_q_entries=300,
            execution_time_seconds=0.5,
            training_records=history_500,
        )

        r2 = ExperimentResult(
            experiment_name="1000 eps",
            episodes=1000,
            max_steps=200,
            learning_rate=0.1,
            discount_factor=0.9,
            final_epsilon=0.01,
            training_success_rate=97.0,
            training_avg_reward=130.0,
            training_avg_steps=30.0,
            training_collision_rate=3.0,
            total_training_collisions=150,
            successful_training_episodes=970,
            recent_100_success_rate=100.0,
            recent_100_avg_reward=170.0,
            recent_100_avg_steps=19.0,
            recent_100_collision_rate=0.2,
            eval_episodes=50,
            eval_success_rate=100.0,
            eval_avg_reward=172.0,
            eval_avg_steps=18.0,
            eval_collision_rate=0.0,
            eval_min_steps=18,
            eval_max_steps=18,
            eval_optimal_route_length=19,
            best_episode=400,
            best_reward=190.0,
            non_zero_q_entries=310,
            execution_time_seconds=0.8,
            training_records=history_1000,
        )

        results = [r1, r2]

        # 1. 4-Panel comparison bar chart
        plot_saved = plot_experiment_comparison(results, save_path=self.plot_bar_path)
        self.assertTrue(plot_saved)
        self.assertTrue(os.path.exists(self.plot_bar_path))
        self.assertGreater(os.path.getsize(self.plot_bar_path), 1000)

        # 2. Learning curves overlay
        curve_saved = plot_learning_curves_comparison(results, save_path=self.plot_curve_path)
        self.assertTrue(curve_saved)
        self.assertTrue(os.path.exists(self.plot_curve_path))
        self.assertGreater(os.path.getsize(self.plot_curve_path), 1000)

    # --------------------------------------------------------------------------
    # 5. Factual Neutral Presentation
    # --------------------------------------------------------------------------
    def test_summary_table_factual_neutrality(self):
        """Verify summary table presents only objective measured values."""
        r1 = ExperimentResult(
            experiment_name="Exp A",
            episodes=500,
            max_steps=200,
            learning_rate=0.1,
            discount_factor=0.9,
            final_epsilon=0.08,
            training_success_rate=94.2,
            training_avg_reward=98.7,
            training_avg_steps=45.5,
            training_collision_rate=8.1,
            total_training_collisions=4062,
            successful_training_episodes=471,
            recent_100_success_rate=100.0,
            recent_100_avg_reward=166.7,
            recent_100_avg_steps=20.8,
            recent_100_collision_rate=0.6,
            eval_episodes=100,
            eval_success_rate=100.0,
            eval_avg_reward=172.0,
            eval_avg_steps=18.0,
            eval_collision_rate=0.0,
            eval_min_steps=18,
            eval_max_steps=18,
            eval_optimal_route_length=19,
            best_episode=203,
            best_reward=184.0,
            non_zero_q_entries=320,
            execution_time_seconds=0.45,
            training_records=[],
        )

        table_text = format_experiment_summary_table([r1])
        self.assertIn("Exp A", table_text)
        self.assertIn("94.2%", table_text)
        self.assertIn("172.0", table_text)
        self.assertIn("18.0", table_text)
        # Verify no unsubstantiated subjective claims like "Winner" or "Best configuration"
        self.assertNotIn("Winner:", table_text)
        self.assertNotIn("Best Model:", table_text)


if __name__ == "__main__":
    unittest.main()
