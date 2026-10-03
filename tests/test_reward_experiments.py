"""
Campus Navigator: Phase 21 Reward Function Experiment Test Suite
================================================================
Module: tests/test_reward_experiments.py

Verifies:
- RewardProfile specifications and default profiles (A, B, C).
- CampusEnvironment reward calculation compliance across all profiles.
- run_single_reward_experiment training and evaluation pipeline.
- Persistence of empirical results to CSV.
- Generation of comparison plots and route overlays.
- Factual neutrality and absence of hardcoded biased conclusions.
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
from environment.campus_env import CampusEnvironment
from experiments.reward_experiments import (
    RewardProfile,
    RewardExperimentResult,
    PROFILE_A,
    PROFILE_B,
    PROFILE_C,
    run_single_reward_experiment,
    plot_reward_profile_comparison,
    plot_reward_learning_curves,
    plot_reward_routes_comparison,
    format_reward_summary_table,
)


class TestRewardExperiments(unittest.TestCase):
    """Test suite covering Phase 21 reward profiles, environment integration, and benchmarking."""

    def setUp(self):
        """Create sandbox directory for test artifacts."""
        self.temp_dir = tempfile.mkdtemp(prefix="reward_exp_test_")
        self.csv_path = os.path.join(self.temp_dir, "test_reward_results.csv")
        self.plot_bar_path = os.path.join(self.temp_dir, "test_reward_comparison.png")
        self.plot_curve_path = os.path.join(self.temp_dir, "test_reward_curves.png")
        self.plot_routes_path = os.path.join(self.temp_dir, "test_reward_routes.png")

    def tearDown(self):
        """Clean up sandbox directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # 1. Reward Profile Specifications
    # --------------------------------------------------------------------------
    def test_standard_profiles_match_specifications(self):
        """Verify standard profiles match the Phase 21 prompt requirements."""
        # Profile A: Simple navigation
        self.assertEqual(PROFILE_A.reward_goal, 100.0)
        self.assertEqual(PROFILE_A.reward_step, -1.0)
        self.assertEqual(PROFILE_A.reward_obstacle, -10.0)
        self.assertFalse(PROFILE_A.use_distance_shaping)

        # Profile B: Efficient navigation
        self.assertEqual(PROFILE_B.reward_goal, 100.0)
        self.assertEqual(PROFILE_B.reward_closer, 5.0)
        self.assertEqual(PROFILE_B.reward_farther, -3.0)
        self.assertEqual(PROFILE_B.reward_step, -1.0)
        self.assertEqual(PROFILE_B.reward_obstacle, -10.0)
        self.assertTrue(PROFILE_B.use_distance_shaping)

        # Profile C: Safety-aware navigation
        self.assertEqual(PROFILE_C.reward_goal, 100.0)
        self.assertEqual(PROFILE_C.reward_closer, 5.0)
        self.assertEqual(PROFILE_C.reward_unsafe, -15.0)
        self.assertEqual(PROFILE_C.reward_obstacle, -20.0)
        self.assertEqual(PROFILE_C.reward_step, -1.0)
        self.assertTrue(PROFILE_C.use_distance_shaping)

    # --------------------------------------------------------------------------
    # 2. CampusEnvironment Reward Calculation Compliance
    # --------------------------------------------------------------------------
    def test_environment_reward_calculation_with_profiles(self):
        """Verify CampusEnvironment dynamically updates reward logic per profile."""
        env = CampusEnvironment(start_pos=(0, 0), goal_pos=(9, 9))

        # Test Profile A
        env.set_reward_profile(PROFILE_A)
        self.assertEqual(env.active_reward_profile_name, PROFILE_A.name)
        # In Profile A, step closer still yields -1.0 because distance shaping is disabled
        r_closer_a = env.calculate_reward(current_state=(0, 0), next_state=(1, 0), is_valid=True)
        self.assertEqual(r_closer_a, -1.0)
        # Goal reward
        r_goal_a = env.calculate_reward(current_state=(8, 9), next_state=(9, 9), is_valid=True)
        self.assertEqual(r_goal_a, 100.0)
        # Obstacle reward
        r_obs_a = env.calculate_reward(current_state=(0, 3), next_state=(0, 4), is_valid=False)
        self.assertEqual(r_obs_a, -10.0)

        # Test Profile B
        env.set_reward_profile(PROFILE_B)
        self.assertEqual(env.active_reward_profile_name, PROFILE_B.name)
        # Moving closer gives +5.0
        r_closer_b = env.calculate_reward(current_state=(0, 0), next_state=(1, 0), is_valid=True)
        self.assertEqual(r_closer_b, 5.0)
        # Moving farther gives -3.0
        r_farther_b = env.calculate_reward(current_state=(1, 0), next_state=(0, 0), is_valid=True)
        self.assertEqual(r_farther_b, -3.0)

        # Test Profile C
        env.set_reward_profile(PROFILE_C)
        self.assertEqual(env.active_reward_profile_name, PROFILE_C.name)
        # Obstacle collision gives -20.0
        r_obs_c = env.calculate_reward(current_state=(0, 0), next_state=(0, 0), is_valid=False)
        self.assertEqual(r_obs_c, -20.0)
        # Unsafe hazard zone gives -15.0
        unsafe_cell = list(env.unsafe_zones)[0]
        r_unsafe_c = env.calculate_reward(current_state=(0, 0), next_state=unsafe_cell, is_valid=True)
        self.assertEqual(r_unsafe_c, -15.0)

    # --------------------------------------------------------------------------
    # 3. Single Profile Execution & Telemetry
    # --------------------------------------------------------------------------
    def test_run_single_reward_experiment_produces_full_metrics(self):
        """Verify run_single_reward_experiment computes all required training and eval metrics."""
        res = run_single_reward_experiment(
            profile=PROFILE_B,
            episodes=30,
            max_steps=50,
            eval_episodes=10,
            verbose=False,
        )

        self.assertIsInstance(res, RewardExperimentResult)
        self.assertEqual(res.profile_name, PROFILE_B.name)
        self.assertEqual(res.episodes, 30)
        self.assertEqual(res.eval_episodes, 10)

        # Telemetry metrics
        self.assertIsInstance(res.training_success_rate, float)
        self.assertIsInstance(res.training_avg_reward, float)
        self.assertIsInstance(res.training_avg_steps, float)
        self.assertIsInstance(res.training_collision_rate, float)
        self.assertIsInstance(res.training_unsafe_visits_rate, float)

        self.assertIsInstance(res.eval_success_rate, float)
        self.assertIsInstance(res.eval_avg_reward, float)
        self.assertIsInstance(res.eval_avg_steps, float)
        self.assertIsInstance(res.eval_collision_rate, float)
        self.assertIsInstance(res.eval_unsafe_visits_rate, float)

        self.assertIsInstance(res.eval_route_path, list)
        self.assertGreater(len(res.eval_route_path), 0)

    # --------------------------------------------------------------------------
    # 4. CSV Persistence and Schema Compliance
    # --------------------------------------------------------------------------
    def test_reward_results_csv_schema_and_persistence(self):
        """Verify reward experiment results serialize cleanly to CSV."""
        r1 = RewardExperimentResult(
            profile_name="Profile A (Simple)",
            description="Simple test",
            episodes=500,
            eval_episodes=100,
            reward_goal=100.0,
            reward_step=-1.0,
            reward_obstacle=-10.0,
            reward_closer=None,
            reward_farther=None,
            reward_unsafe=-1.0,
            use_distance_shaping=False,
            training_success_rate=95.0,
            training_avg_reward=15.0,
            training_avg_steps=35.0,
            training_collision_rate=5.0,
            training_unsafe_visits_rate=1.5,
            total_training_collisions=2500,
            total_training_unsafe_visits=750,
            recent_100_avg_reward=80.0,
            recent_100_avg_steps=19.0,
            recent_100_success_rate=100.0,
            eval_success_rate=100.0,
            eval_avg_reward=83.0,
            eval_avg_steps=18.0,
            eval_collision_rate=0.0,
            eval_unsafe_visits_rate=1.0,
            eval_route_length=19,
            eval_route_path=[(0, 0), (1, 0), (9, 9)],
            execution_time_seconds=0.5,
        )

        df = pd.DataFrame([r1.to_dict()])
        df.to_csv(self.csv_path, index=False)

        self.assertTrue(os.path.exists(self.csv_path))
        loaded_df = pd.read_csv(self.csv_path)
        self.assertEqual(len(loaded_df), 1)

        required_cols = [
            "profile_name",
            "training_success_rate",
            "training_avg_reward",
            "training_avg_steps",
            "training_collision_rate",
            "training_unsafe_visits_rate",
            "eval_success_rate",
            "eval_avg_reward",
            "eval_avg_steps",
            "eval_collision_rate",
            "eval_unsafe_visits_rate",
        ]
        for col in required_cols:
            self.assertIn(col, loaded_df.columns)

    # --------------------------------------------------------------------------
    # 5. Plot Generation
    # --------------------------------------------------------------------------
    def test_plot_generation_creates_valid_images(self):
        """Verify comparison plots and route maps are generated without errors."""
        history = [{"episode": i, "steps": 30, "total_reward": 50 + i} for i in range(20)]
        res_a = RewardExperimentResult(
            profile_name="Profile A (Simple)",
            description="Simple",
            episodes=20,
            eval_episodes=10,
            reward_goal=100.0,
            reward_step=-1.0,
            reward_obstacle=-10.0,
            reward_closer=None,
            reward_farther=None,
            reward_unsafe=-1.0,
            use_distance_shaping=False,
            training_success_rate=90.0,
            training_avg_reward=20.0,
            training_avg_steps=30.0,
            training_collision_rate=4.0,
            training_unsafe_visits_rate=1.2,
            total_training_collisions=80,
            total_training_unsafe_visits=24,
            recent_100_avg_reward=20.0,
            recent_100_avg_steps=30.0,
            recent_100_success_rate=90.0,
            eval_success_rate=100.0,
            eval_avg_reward=83.0,
            eval_avg_steps=18.0,
            eval_collision_rate=0.0,
            eval_unsafe_visits_rate=1.0,
            eval_route_length=19,
            eval_route_path=[(0, 0), (1, 0), (2, 0), (9, 9)],
            execution_time_seconds=0.2,
            training_records=history,
        )

        res_b = RewardExperimentResult(
            profile_name="Profile B (Efficient)",
            description="Efficient",
            episodes=20,
            eval_episodes=10,
            reward_goal=100.0,
            reward_step=-1.0,
            reward_obstacle=-10.0,
            reward_closer=5.0,
            reward_farther=-3.0,
            reward_unsafe=-10.0,
            use_distance_shaping=True,
            training_success_rate=95.0,
            training_avg_reward=120.0,
            training_avg_steps=25.0,
            training_collision_rate=3.0,
            training_unsafe_visits_rate=1.0,
            total_training_collisions=60,
            total_training_unsafe_visits=20,
            recent_100_avg_reward=120.0,
            recent_100_avg_steps=25.0,
            recent_100_success_rate=95.0,
            eval_success_rate=100.0,
            eval_avg_reward=172.0,
            eval_avg_steps=18.0,
            eval_collision_rate=0.0,
            eval_unsafe_visits_rate=1.0,
            eval_route_length=19,
            eval_route_path=[(0, 0), (0, 1), (0, 2), (9, 9)],
            execution_time_seconds=0.2,
            training_records=history,
        )

        results = [res_a, res_b]

        # 1. 4-Panel comparison bar chart
        saved_bar = plot_reward_profile_comparison(results, save_path=self.plot_bar_path, verbose=False)
        self.assertTrue(os.path.exists(saved_bar))
        self.assertGreater(os.path.getsize(saved_bar), 1000)

        # 2. Learning curves overlay
        saved_curves = plot_reward_learning_curves(results, save_path=self.plot_curve_path, verbose=False)
        self.assertTrue(os.path.exists(saved_curves))
        self.assertGreater(os.path.getsize(saved_curves), 1000)

        # 3. Routes visualization
        saved_routes = plot_reward_routes_comparison(results, save_path=self.plot_routes_path, verbose=False)
        self.assertTrue(os.path.exists(saved_routes))
        self.assertGreater(os.path.getsize(saved_routes), 1000)

    # --------------------------------------------------------------------------
    # 6. Factual Neutral Presentation
    # --------------------------------------------------------------------------
    def test_summary_table_factual_neutrality(self):
        """Verify summary table presents only objective measured values."""
        r1 = RewardExperimentResult(
            profile_name="Profile A (Simple)",
            description="Simple",
            episodes=800,
            eval_episodes=100,
            reward_goal=100.0,
            reward_step=-1.0,
            reward_obstacle=-10.0,
            reward_closer=None,
            reward_farther=None,
            reward_unsafe=-1.0,
            use_distance_shaping=False,
            training_success_rate=96.6,
            training_avg_reward=14.3,
            training_avg_steps=35.4,
            training_collision_rate=5.3,
            training_unsafe_visits_rate=1.5,
            total_training_collisions=4214,
            total_training_unsafe_visits=1158,
            recent_100_avg_reward=79.4,
            recent_100_avg_steps=19.0,
            recent_100_success_rate=100.0,
            eval_success_rate=100.0,
            eval_avg_reward=83.0,
            eval_avg_steps=18.0,
            eval_collision_rate=0.0,
            eval_unsafe_visits_rate=1.0,
            eval_route_length=19,
            eval_route_path=[(0, 0)],
            execution_time_seconds=0.57,
        )

        table_text = format_reward_summary_table([r1])
        self.assertIn("Profile A (Simple)", table_text)
        self.assertIn("96.6%", table_text)
        self.assertIn("83.0", table_text)
        self.assertIn("18.0", table_text)
        # Ensure no biased subjective assertions
        self.assertNotIn("Winner:", table_text)
        self.assertNotIn("Best Configuration:", table_text)


if __name__ == "__main__":
    unittest.main()
