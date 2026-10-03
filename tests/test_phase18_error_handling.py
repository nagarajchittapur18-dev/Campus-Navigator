"""
Campus Navigator: Phase 18 Error Handling & Resilience Test Suite
================================================================
Comprehensive verification for all 10 major failure scenarios:
1. Missing Q-table
2. Missing result CSV
3. Corrupted Q-table (0-byte, invalid format, NaN/Inf, shape mismatch)
4. Invalid configuration (malformed JSON, negative lr, discount > 1, bad episodes)
5. Missing assets (plots/fonts, plot rendering fallback)
6. Training interruption (graceful KeyboardInterrupt handling)
7. Invalid action (out-of-bounds, negative, non-int)
8. Invalid state (out-of-bounds start/destination coords, index out-of-range)
9. Pygame initialization failure (headless/no-display recovery)
10. Empty result dataset (0-byte CSV, empty DataFrame, < 2 rows)
11. Logging subsystem verification (logs/campus_navigator.log write verification)
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pygame

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from utils.logger import setup_logger, log_info, log_warning, log_error, log_exception
from agent.q_learning import QLearningAgent
from environment.campus_env import CampusEnvironment
from visualization.plots import load_training_data, generate_all_plots
from training.train import train_agent, train_multi_destinations


class TestPhase18ErrorHandling(unittest.TestCase):
    """Test suite covering Phase 18 resilience, error handling, and logging."""

    @classmethod
    def setUpClass(cls):
        """Set up temporary sandbox directory and headless pygame."""
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        cls.temp_dir = tempfile.mkdtemp(prefix="campus_nav_test_p18_")

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary sandbox directory and quit pygame."""
        pygame.quit()
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # Scenario 1: Missing Q-Table
    # --------------------------------------------------------------------------
    def test_01_missing_q_table_handled_gracefully(self):
        """Scenario 1: Loading a non-existent Q-table must return False without crashing."""
        agent = QLearningAgent(rows=10, cols=10, num_actions=4)
        missing_path = os.path.join(self.temp_dir, "nonexistent_q_table.npy")

        # Must not raise FileNotFoundError or unhandled exception
        success = agent.load_model(missing_path)
        self.assertFalse(success)
        self.assertEqual(agent.last_load_status, "MISSING")
        self.assertIn("no trained model", agent.last_load_error_msg.lower())

    # --------------------------------------------------------------------------
    # Scenario 2: Missing Result CSV
    # --------------------------------------------------------------------------
    def test_02_missing_result_csv_handled_gracefully(self):
        """Scenario 2: Loading a missing results CSV must return None and clear error message."""
        missing_csv = os.path.join(self.temp_dir, "nonexistent_results.csv")
        df, err = load_training_data(missing_csv)
        self.assertIsNone(df)
        self.assertIsNotNone(err)
        self.assertIn("not found", err.lower())

    # --------------------------------------------------------------------------
    # Scenario 3: Corrupted Q-Table
    # --------------------------------------------------------------------------
    def test_03a_corrupted_q_table_empty_file(self):
        """Scenario 3a: A 0-byte Q-table file must be detected and rejected."""
        agent = QLearningAgent(rows=10, cols=10, num_actions=4)
        empty_path = os.path.join(self.temp_dir, "empty_q_table.npy")
        with open(empty_path, "wb") as f:
            pass  # 0 bytes

        success = agent.load_model(empty_path)
        self.assertFalse(success)
        self.assertEqual(agent.last_load_status, "EMPTY_FILE")
        self.assertIn("empty", agent.last_load_error_msg.lower())

    def test_03b_corrupted_q_table_invalid_bytes(self):
        """Scenario 3b: Corrupted random non-NumPy bytes must be rejected gracefully."""
        agent = QLearningAgent(rows=10, cols=10, num_actions=4)
        garbage_path = os.path.join(self.temp_dir, "garbage_q_table.npy")
        with open(garbage_path, "wb") as f:
            f.write(b"NOT_A_NUMPY_FILE_CORRUPTED_BYTES_12345")

        success = agent.load_model(garbage_path)
        self.assertFalse(success)
        self.assertEqual(agent.last_load_status, "CORRUPTED")
        self.assertIn("failed", agent.last_load_error_msg.lower())

    def test_03c_corrupted_q_table_nan_inf_values(self):
        """Scenario 3c: Q-table containing NaN or Inf values must be rejected."""
        agent = QLearningAgent(rows=10, cols=10, num_actions=4)
        
        # Test NaN
        nan_path = os.path.join(self.temp_dir, "nan_q_table.npy")
        nan_table = np.zeros((100, 4), dtype=np.float32)
        nan_table[10, 2] = np.nan
        np.save(nan_path, nan_table)

        success = agent.load_model(nan_path)
        self.assertFalse(success)
        self.assertEqual(agent.last_load_status, "CORRUPTED_VALUES")
        self.assertIn("nan", agent.last_load_error_msg.lower())

        # Test Inf
        inf_path = os.path.join(self.temp_dir, "inf_q_table.npy")
        inf_table = np.zeros((100, 4), dtype=np.float32)
        inf_table[5, 1] = np.inf
        np.save(inf_path, inf_table)

        success_inf = agent.load_model(inf_path)
        self.assertFalse(success_inf)
        self.assertEqual(agent.last_load_status, "CORRUPTED_VALUES")
        self.assertIn("inf", agent.last_load_error_msg.lower())

    def test_03d_corrupted_q_table_invalid_shape(self):
        """Scenario 3d: Q-table with mismatched dimensions must be rejected."""
        agent = QLearningAgent(rows=10, cols=10, num_actions=4)
        wrong_shape_path = os.path.join(self.temp_dir, "wrong_shape_q_table.npy")
        wrong_table = np.zeros((50, 2), dtype=np.float32)  # Expected (100, 4)
        np.save(wrong_shape_path, wrong_table)

        success = agent.load_model(wrong_shape_path)
        self.assertFalse(success)
        self.assertEqual(agent.last_load_status, "INVALID_SHAPE")
        self.assertIn("shape", agent.last_load_error_msg.lower())

    # --------------------------------------------------------------------------
    # Scenario 4: Invalid Configuration
    # --------------------------------------------------------------------------
    def test_04a_invalid_config_validation_rejection(self):
        """Scenario 4a: validate_config must catch and reject all invalid values."""
        # Zero actions
        valid, msg = config.validate_config({"num_actions": 0})
        self.assertFalse(valid)
        self.assertIn("zero actions", msg.lower())

        # Negative learning rate
        valid, msg = config.validate_config({"learning_rate": -0.05})
        self.assertFalse(valid)
        self.assertIn("learning rate", msg.lower())

        # Discount factor > 1
        valid, msg = config.validate_config({"discount_factor": 1.25})
        self.assertFalse(valid)
        self.assertIn("discount factor", msg.lower())

        # Negative episodes
        valid, msg = config.validate_config({"training_episodes": -100})
        self.assertFalse(valid)
        self.assertIn("episodes", msg.lower())

        # Zero or negative max steps
        valid, msg = config.validate_config({"max_steps_per_episode": 0})
        self.assertFalse(valid)
        self.assertIn("steps", msg.lower())

        # Min epsilon > initial epsilon
        valid, msg = config.validate_config({"initial_epsilon": 0.05, "min_epsilon": 0.20})
        self.assertFalse(valid)
        self.assertIn("minimum epsilon", msg.lower())

    def test_04b_corrupted_json_config_recovers_gracefully(self):
        """Scenario 4b: A corrupted/malformed config.json must fall back to defaults safely."""
        corrupt_config_path = os.path.join(self.temp_dir, "corrupt_config.json")
        with open(corrupt_config_path, "w", encoding="utf-8") as f:
            f.write("{ INVALID JSON SYNTAX ,,, missing brackets")

        loaded = config.load_config(corrupt_config_path)
        self.assertIsInstance(loaded, dict)
        # Should have fallen back to default learning rate
        self.assertEqual(loaded["learning_rate"], config.DEFAULT_CONFIG["learning_rate"])
        self.assertEqual(loaded["training_episodes"], config.DEFAULT_CONFIG["training_episodes"])

    # --------------------------------------------------------------------------
    # Scenario 5: Missing Assets (Plots / Images)
    # --------------------------------------------------------------------------
    def test_05_missing_plots_dashboard_fallback(self):
        """Scenario 5: When plot images do not exist on disk, menu handles None without crash."""
        from game.menu import GameMenu
        screen = pygame.display.set_mode((1100, 750))
        menu = GameMenu(screen)

        # Mock results dir to point to empty temp directory
        empty_results_dir = os.path.join(self.temp_dir, "empty_results")
        os.makedirs(empty_results_dir, exist_ok=True)

        with patch("config.RESULTS_DIR", empty_results_dir), \
             patch("config.TRAINING_RESULTS_SAVE_PATH", os.path.join(empty_results_dir, "nonexistent.csv")):
            menu._load_results_plots(force_reload=True)
            # All cached plots should safely be None without raising uncaught exceptions
            for k in ["reward_vs_episode", "steps_vs_episode", "success_rate", "collision_rate"]:
                self.assertIsNone(menu.results_cached_plots.get(k))

            # Test _draw_plot_card with None surface does not crash
            menu._draw_plot_card(50, 50, 450, 250, "Test Missing Plot", None, (37, 99, 235))

    # --------------------------------------------------------------------------
    # Scenario 6: Training Interruption
    # --------------------------------------------------------------------------
    def test_06_training_interruption_keyboard_interrupt(self):
        """Scenario 6: KeyboardInterrupt during training must be caught and save progress safely."""
        env = CampusEnvironment(rows=10, cols=10)
        agent = QLearningAgent(rows=10, cols=10, num_actions=4)

        custom_q_path = os.path.join(self.temp_dir, "interrupted_q_table.npy")
        custom_multi_path = os.path.join(self.temp_dir, "interrupted_q_table_multi.npy")
        custom_csv_path = os.path.join(self.temp_dir, "interrupted_results.csv")

        # Mock env.reset to simulate 2 successful episodes and interrupt at episode 3
        original_reset = env.reset
        reset_count = [0]

        def interrupt_reset(*args, **kwargs):
            reset_count[0] += 1
            if reset_count[0] >= 3:
                raise KeyboardInterrupt("Simulated Ctrl+C by user")
            return original_reset(*args, **kwargs)

        with patch.object(env, "reset", side_effect=interrupt_reset), \
             patch("config.Q_TABLE_SAVE_PATH", custom_q_path), \
             patch("config.Q_TABLE_MULTI_SAVE_PATH", custom_multi_path), \
             patch("config.TRAINING_RESULTS_PATH", custom_csv_path), \
             patch("config.TRAINING_RESULTS_SAVE_PATH", custom_csv_path):
            
            # Must not crash or raise KeyboardInterrupt to caller
            trained_agent, df = train_agent(
                env=env,
                agent=agent,
                episodes=10,
                max_steps=5,
                verbose=False,
                save_model=True,
                save_results=True,
            )

            self.assertIsNotNone(trained_agent)
            self.assertEqual(len(df), 2)
            # Partial weights and metrics must be saved
            self.assertTrue(os.path.exists(custom_q_path))
            self.assertTrue(os.path.exists(custom_csv_path))

    # --------------------------------------------------------------------------
    # Scenario 7: Invalid Action
    # --------------------------------------------------------------------------
    def test_07_invalid_action_handled_safely(self):
        """Scenario 7: Negative, out-of-bounds, or non-integer actions must not crash the env or agent."""
        env = CampusEnvironment(rows=10, cols=10)
        env.reset()
        initial_state = env.agent_pos

        # Negative action
        next_state, reward, done, info = env.step(-1)
        self.assertEqual(next_state, initial_state)
        self.assertEqual(reward, config.REWARD_INVALID_MOVE)

        # Out-of-bounds action
        next_state, reward, done, info = env.step(999)
        self.assertEqual(next_state, initial_state)

        # String action
        next_state, reward, done, info = env.step("INVALID_ACTION")  # type: ignore
        self.assertEqual(next_state, initial_state)

        # Agent update with invalid action must not crash with IndexError
        agent = QLearningAgent(rows=10, cols=10, num_actions=4)
        td_neg = agent.update(state=5, action=-1, reward=1.0, next_state=6, done=False)
        self.assertEqual(td_neg, 0.0)
        td_huge = agent.update(state=5, action=999, reward=1.0, next_state=6, done=False)
        self.assertEqual(td_huge, 0.0)

    # --------------------------------------------------------------------------
    # Scenario 8: Invalid State
    # --------------------------------------------------------------------------
    def test_08_invalid_state_and_coordinates_handled_safely(self):
        """Scenario 8: Out-of-bounds coordinates and states must be rejected cleanly."""
        env = CampusEnvironment(rows=10, cols=10)

        # Out-of-bounds start raises ValueError
        with self.assertRaises(ValueError):
            env.set_start((-1, 5))
        with self.assertRaises(ValueError):
            env.set_start((100, 100))
        with self.assertRaises(ValueError):
            env.set_start((0, -2))

        # Out-of-bounds destination raises ValueError
        with self.assertRaises(ValueError):
            env.set_destination((-5, 0))
        with self.assertRaises(ValueError):
            env.set_destination((50, 50))

        # Agent state index out-of-bounds protection
        agent = QLearningAgent(rows=10, cols=10, num_actions=4)
        with self.assertRaises(ValueError):
            agent.state_to_index((-1, 5))
        with self.assertRaises(ValueError):
            agent.state_to_index((100, 100))

        with self.assertRaises(ValueError):
            agent.index_to_state(-5)
        with self.assertRaises(ValueError):
            agent.index_to_state(9999)

    # --------------------------------------------------------------------------
    # Scenario 9: Pygame Initialization Failure
    # --------------------------------------------------------------------------
    def test_09_pygame_init_failure_handled(self):
        """Scenario 9: Display initialization failures must raise descriptive RuntimeError."""
        from game.game import CampusNavigatorGame

        with patch("pygame.display.set_mode", side_effect=pygame.error("No available video device")):
            with self.assertRaises(RuntimeError) as ctx:
                CampusNavigatorGame()
            
            err_msg = str(ctx.exception)
            self.assertIn("Pygame graphical initialization failed", err_msg)

    # --------------------------------------------------------------------------
    # Scenario 10: Empty Result Dataset
    # --------------------------------------------------------------------------
    def test_10a_empty_result_dataset_0_bytes(self):
        """Scenario 10a: 0-byte result CSV must be detected and return friendly error."""
        empty_csv = os.path.join(self.temp_dir, "zero_byte_results.csv")
        with open(empty_csv, "wb") as f:
            pass

        df, err = load_training_data(empty_csv)
        self.assertIsNone(df)
        self.assertIsNotNone(err)
        self.assertIn("empty", err.lower())

    def test_10b_empty_result_dataset_header_only(self):
        """Scenario 10b: Header-only result CSV (0 episodes) must be detected."""
        header_only_csv = os.path.join(self.temp_dir, "header_only_results.csv")
        with open(header_only_csv, "w", encoding="utf-8") as f:
            f.write("episode,steps,total_reward,success,collisions,epsilon\n")

        df, err = load_training_data(header_only_csv)
        self.assertIsNone(df)
        self.assertIsNotNone(err)
        self.assertIn("empty", err.lower())

    def test_10c_single_row_result_dataset(self):
        """Scenario 10c: CSV with only 1 episode must report insufficient data for plotting."""
        single_row_csv = os.path.join(self.temp_dir, "single_row_results.csv")
        with open(single_row_csv, "w", encoding="utf-8") as f:
            f.write("episode,steps,total_reward,success,collisions,epsilon\n")
            f.write("1,15,85.0,1,0,0.99\n")

        df, err = load_training_data(single_row_csv)
        self.assertIsNone(df)
        self.assertIsNotNone(err)
        self.assertIn("at least 2", err.lower())

    # --------------------------------------------------------------------------
    # Logging System Verification
    # --------------------------------------------------------------------------
    def test_11_logging_subsystem_writes_file(self):
        """Verification that logs/campus_navigator.log is created and records messages."""
        test_log_msg = "Phase 18 Resilience Verification Event - UniqueID 987654321"
        log_info(test_log_msg)

        log_path = config.LOG_FILE_PATH
        self.assertTrue(os.path.exists(log_path), f"Log file does not exist at {log_path}")

        with open(log_path, "r", encoding="utf-8") as f:
            contents = f.read()

        self.assertIn(test_log_msg, contents)
        self.assertIn("[INFO]", contents)


if __name__ == "__main__":
    unittest.main()
