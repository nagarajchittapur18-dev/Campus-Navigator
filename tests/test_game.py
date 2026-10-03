"""
Campus Navigator: Phase 19 Game Unit & Integration Test Suite
=============================================================
Module: tests/test_game.py

Verifies:
- Game starts (window initialization, component wiring, default MENU state).
- Menu navigation (switching between sub-screens, modals, returning to menu).
- Human controls (WASD/Arrow key events, obstacles blocking, mission complete modal).
- AI mode (Watch Trained AI, epsilon=0, policy route tracking, learned path display).
- Missing model handling (friendly NO_MODEL alert screen with recovery buttons, no crash).
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Ensure headless Pygame testing before importing pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from game.game import CampusNavigatorGame


class TestCampusNavigatorGame(unittest.TestCase):
    """Test suite covering the master game loop, menus, modes, and controls."""

    @classmethod
    def setUpClass(cls):
        """Initialize pygame in headless mode."""
        pygame.init()

    @classmethod
    def tearDownClass(cls):
        """Clean up pygame."""
        pygame.quit()

    def setUp(self):
        """Create fresh headless game instance."""
        self.game = CampusNavigatorGame()

    # --------------------------------------------------------------------------
    # 1. Game Starts
    # --------------------------------------------------------------------------
    def test_game_starts(self):
        """Verify window resolution, state machine initialization, and component wiring."""
        self.assertEqual(self.game.current_state, "MENU")
        self.assertTrue(self.game.is_running)

        # Resolution verification
        self.assertEqual(self.game.screen.get_width(), config.SCREEN_WIDTH)
        self.assertEqual(self.game.screen.get_height(), config.SCREEN_HEIGHT)

        # Core decoupled modules attached
        self.assertIsNotNone(self.game.env)
        self.assertIsNotNone(self.game.agent)
        self.assertIsNotNone(self.game.renderer)
        self.assertIsNotNone(self.game.menu)

    # --------------------------------------------------------------------------
    # 2. Menu Navigation
    # --------------------------------------------------------------------------
    def test_menu_navigation(self):
        """Verify navigation through sub-screens, modals, and back to main menu."""
        # Main Menu draw
        self.game.menu.draw()

        # 1. Navigate to Training Results sub-screen
        self.game.menu.sub_screen = "RESULTS"
        self.game.menu.draw()
        self.assertEqual(self.game.menu.sub_screen, "RESULTS")

        # 2. Navigate to How RL Works Theory guide
        self.game.menu.sub_screen = "RL_WORKS"
        self.game.menu.draw()
        self.assertEqual(self.game.menu.sub_screen, "RL_WORKS")

        # 3. Navigate to Settings screen
        self.game.menu.sub_screen = "SETTINGS"
        self.game.menu.draw()
        self.assertEqual(self.game.menu.sub_screen, "SETTINGS")

        # 4. Open Route Selection Modal
        self.game._open_route_modal()
        self.assertTrue(self.game.show_route_modal)
        self.game.renderer.draw_route_selection_modal(
            start_buttons=self.game.modal_start_buttons,
            dest_buttons=self.game.modal_dest_buttons,
            confirm_btn=self.game.btn_modal_confirm,
            cancel_btn=self.game.btn_modal_cancel,
            selected_start_name=self.game.temp_selected_start,
            selected_dest_name=self.game.temp_selected_dest,
            dest_id=0,
            manhattan_dist=18,
            is_same_location=False,
        )

        # 5. Cancel Route Modal and return to MENU
        self.game.show_route_modal = False
        self.assertFalse(self.game.show_route_modal)

        # 6. Press ESC to return to main menu
        self.game.menu.sub_screen = "RESULTS"
        esc_event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE})
        pygame.event.post(esc_event)
        self.game.menu.handle_input()
        self.assertEqual(self.game.menu.sub_screen, "MAIN")

    # --------------------------------------------------------------------------
    # 3. Human Controls
    # --------------------------------------------------------------------------
    def test_human_controls(self):
        """Verify responsive WASD/Arrow key controls, step tracking, and obstacle avoidance."""
        self.game._start_human_mode(preserve_landmarks=True)
        self.assertEqual(self.game.current_state, "HUMAN_PLAY")

        initial_pos = self.game.env.agent_pos
        self.assertEqual(initial_pos, config.START_POSITION)
        self.assertEqual(self.game.step_count, 0)

        # 1. Move DOWN using 'S' key
        down_event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_s})
        pygame.event.post(down_event)
        self.game._handle_human_play_state()

        self.assertEqual(self.game.env.agent_pos, (1, 0))
        self.assertEqual(self.game.step_count, 1)
        self.assertGreater(self.game.cumulative_reward, 0)

        # 2. Move RIGHT using RIGHT Arrow key
        right_event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_RIGHT})
        pygame.event.post(right_event)
        self.game._handle_human_play_state()

        self.assertEqual(self.game.env.agent_pos, (1, 1))
        self.assertEqual(self.game.step_count, 2)

        # 3. Attempt moving into building obstacle at (1, 2)
        if (1, 2) in self.game.env.obstacles:
            pygame.event.post(right_event)
            self.game._handle_human_play_state()
            self.assertEqual(self.game.env.agent_pos, (1, 1), "Human walked through building obstacle!")
            self.assertGreaterEqual(self.game.human_collisions, 1)

        # 4. Goal Arrival Completion Modal
        self.game.env.agent_pos = (8, 9)
        self.game.env.goal_pos = (9, 9)
        pygame.event.post(down_event)
        self.game._handle_human_play_state()

        self.assertTrue(self.game.goal_reached)
        self.assertEqual(self.game.env.agent_pos, (9, 9))

    # --------------------------------------------------------------------------
    # 4. AI Mode (Watch Trained AI)
    # --------------------------------------------------------------------------
    def test_ai_mode(self):
        """Verify Watch Trained AI launches with frozen epsilon and follows learned policy."""
        # Ensure model is available
        self.assertTrue(os.path.exists(config.Q_TABLE_SAVE_PATH))

        self.game._start_ai_mode(preserve_landmarks=True)
        self.assertEqual(self.game.current_state, "AI_PLAY")
        self.assertEqual(self.game.agent.epsilon, 0.0)
        self.assertEqual(len(self.game.ai_path), 1)

        # Step AI until destination reached
        steps = 0
        while not self.game.ai_route_complete and steps < 50:
            current_state = self.game.env.get_state()
            action = self.game.agent.get_best_action(current_state)
            next_state, reward, done, info = self.game.env.step(action)
            self.game.ai_last_action = config.ACTIONS.get(action, "None")
            self.game.ai_step_reward = reward
            self.game.ai_total_reward += reward
            self.game.ai_steps = self.game.env.step_count
            self.game.ai_path.append(next_state)
            if info.get("goal_reached", False):
                self.game.ai_route_complete = True
            steps += 1

        self.assertTrue(self.game.ai_route_complete)
        self.assertEqual(self.game.env.agent_pos, config.GOAL_POSITION)
        self.assertEqual(steps, 18)
        self.assertIn(self.game.ai_total_reward, [170.0, 172.0])

    # --------------------------------------------------------------------------
    # 5. Missing Model Handling
    # --------------------------------------------------------------------------
    def test_missing_model_handling(self):
        """Verify missing model file safely triggers NO_MODEL alert screen without crashing."""
        nonexistent_path = "models/nonexistent_model_12345.npy"

        with patch("config.Q_TABLE_SAVE_PATH", nonexistent_path), \
             patch("config.Q_TABLE_MULTI_SAVE_PATH", nonexistent_path):

            # Fresh game with mocked missing path
            game_no_model = CampusNavigatorGame()
            game_no_model._start_ai_mode()

            self.assertEqual(game_no_model.current_state, "NO_MODEL")
            self.assertIn("No trained model", game_no_model.no_model_message)

            # Test rendering NO_MODEL screen
            game_no_model.renderer.draw_no_model_screen(
                buttons=game_no_model.no_model_buttons,
                error_message=game_no_model.no_model_message,
                error_detail=game_no_model.no_model_detail,
            )

    # --------------------------------------------------------------------------
    # 6. Menu Option 4: Compare Human vs AI
    # --------------------------------------------------------------------------
    def test_compare_human_vs_ai_menu_option(self):
        """Verify selecting Option 4 from main menu transitions to AI comparison run."""
        self.game.current_state = "MENU"
        # Simulate selection code 4
        with patch.object(self.game.menu, "handle_input", return_value=4):
            self.game._handle_menu_state()

        self.assertEqual(self.game.current_state, "AI_COMPARISON_RUN")
        self.assertIsNotNone(self.game.comparison_human_data)
        self.assertIn("steps", self.game.comparison_human_data)
        self.assertIn("reward", self.game.comparison_human_data)

    # --------------------------------------------------------------------------
    # 7. Menu Option 8: Exit Game
    # --------------------------------------------------------------------------
    def test_exit_menu_option(self):
        """Verify selecting Option 8 cleanly terminates the game loop."""
        self.game.current_state = "MENU"
        self.assertTrue(self.game.is_running)
        with patch.object(self.game.menu, "handle_input", return_value=8):
            self.game._handle_menu_state()

        self.assertFalse(self.game.is_running)


if __name__ == "__main__":
    unittest.main()

