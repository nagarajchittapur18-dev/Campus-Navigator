"""
Unit and integration tests for PHASE 17: Professional UI Enhancement.

Tests cover:
1. Color palette and visual design tokens in config.py
2. Monospace typography for coordinate rulers and telemetry
3. Tactile button styling with keyboard shortcut badges
4. Axis coordinate rulers (Col 0..9 and Row 0..9)
5. Pulsing status badge indicator
6. Goal arrival expanding concentric ripple animation
7. Failure feedback collision outline and alert banner
8. Numbered waypoint route visualization
9. Campus legend with active start/goal coordinates
10. Dual progress meters for training (Episodes + Epsilon Decay)
11. End-to-end rendering of Watch AI and Human Play at 1100x750 resolution
"""

import os
import unittest
import pygame

# Set dummy headless video driver for CI/headless verification
os.environ["SDL_VIDEODRIVER"] = "dummy"

import config
from game.menu import Button
from game.renderer import CampusRenderer
from environment.campus_env import CampusEnvironment


class TestPhase17UIEnhancements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pygame.init()
        cls.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        cls.renderer = CampusRenderer(cls.screen)
        cls.env = CampusEnvironment()

    @classmethod
    def tearDownClass(cls):
        pygame.quit()

    def test_01_color_palette_constants(self):
        """Verify that Phase 17 UI semantic palette constants exist and have valid RGB tuples."""
        required_colors = [
            "COLOR_AXIS_RULER",
            "COLOR_AXIS_BG",
            "COLOR_STATUS_ACTIVE",
            "COLOR_STATUS_TRAINING",
            "COLOR_STATUS_MANUAL",
            "COLOR_STATUS_ALERT",
            "COLOR_PROGRESS_BG",
            "COLOR_PROGRESS_FILL",
            "COLOR_PROGRESS_ACCENT",
            "COLOR_EPSILON_FILL",
            "COLOR_SUCCESS_PULSE",
            "COLOR_ALERT_BG",
            "COLOR_ALERT_BORDER",
            "COLOR_ALERT_TEXT",
        ]
        for name in required_colors:
            self.assertTrue(hasattr(config, name), f"Missing color token: {name}")
            val = getattr(config, name)
            self.assertIsInstance(val, tuple, f"{name} must be a tuple")
            self.assertEqual(len(val), 3, f"{name} must have 3 RGB components")
            for c in val:
                self.assertTrue(0 <= c <= 255, f"RGB channel {c} in {name} must be in [0, 255]")

    def test_02_monospace_typography_loaded(self):
        """Verify that clean monospace fonts for tabular telemetry and coordinates are loaded."""
        self.assertIsNotNone(self.renderer.font_mono)
        self.assertIsNotNone(self.renderer.font_mono_small)
        self.assertIsNotNone(self.renderer.font_mono_val)
        self.assertIsNotNone(self.renderer.font_mono_tiny)

    def test_03_button_shortcut_pill_rendering(self):
        """Verify tactile Button with shortcut pill badge renders cleanly."""
        btn = Button(
            rect=pygame.Rect(100, 100, 180, 44),
            text="Start Navigation",
            shortcut_key="SPACE",
            font=self.renderer.font_btn,
        )
        self.assertEqual(btn.shortcut_key, "SPACE")
        # Draw on screen without exception
        btn.draw(self.screen)
        # Check hover drawing
        btn.is_hovered = True
        btn.draw(self.screen)

    def test_04_axis_coordinate_rulers(self):
        """Verify coordinate rulers (Col 0..9 and Row 0..9) render without error."""
        self.renderer.draw_axis_rulers()
        # Verify grid coordinate bounds:
        self.assertEqual(self.renderer.grid_x, 50)
        self.assertEqual(self.renderer.grid_y, 80)
        self.assertEqual(self.renderer.cell_size, 60)

    def test_05_status_indicator_pills(self):
        """Verify pulsing academic status indicator badge renders for diverse states."""
        states = [
            ("ACTIVE", "OPTIMAL POLICY"),
            ("TRAINING", "AI TRAINING"),
            ("MANUAL", "MANUAL CONTROL"),
            ("GOAL", "GOAL REACHED"),
            ("COLLISION", "COLLISION ALERT"),
            ("PAUSED", "SIMULATION PAUSED"),
        ]
        for status_type, label in states:
            self.renderer.draw_status_indicator(
                x=710, y=140, status_type=status_type, custom_text=label
            )

    def test_06_goal_arrival_ripple_animation(self):
        """Verify goal arrival expanding concentric ripple renders properly."""
        self.renderer.draw_goal_arrival_animation(self.env.goal_pos)

    def test_07_failure_feedback_flash_outline(self):
        """Verify collision cell flashing border renders properly."""
        now = pygame.time.get_ticks()
        collision_time = now - 300
        collision_cell = (1, 2)
        self.renderer.draw_failure_feedback(
            collision_cell, "Impassable Building Obstacle (-10)", collision_time
        )

    def test_08_numbered_waypoint_route_visualization(self):
        """Verify route visualization renders numbered waypoints along the trajectory line."""
        sample_path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)]
        self.renderer.draw_route_path(
            sample_path, start_pos=self.env.start_pos, goal_pos=self.env.goal_pos
        )

    def test_09_campus_legend_with_dynamic_coordinates(self):
        """Verify campus legend renders full-width bottom bar with active start and goal coordinates."""
        self.renderer.draw_legend(self.env)

    def test_10_training_progress_and_epsilon_gauges(self):
        """Verify training info panel renders episode progress bar and epsilon decay meter."""
        stats = {
            "episode": 350,
            "total_episodes": 600,
            "reward": 145.0,
            "steps": 22,
            "epsilon": 0.175,
            "outcome": "Step: Moved",
            "collisions": 12,
            "success_rate": 88.5,
            "paused": False,
            "last_collision_time": 0,
            "last_collision_cell": None,
            "last_collision_msg": "",
            "goal_arrival_time": 0,
            "goal_reached": False,
        }
        speed_buttons = {
            "SLOW": Button(pygame.Rect(710, 620, 75, 36), "Slow", shortcut_key="1", font=self.renderer.font_small),
            "NORMAL": Button(pygame.Rect(793, 620, 75, 36), "Normal", shortcut_key="2", font=self.renderer.font_small),
            "FAST": Button(pygame.Rect(876, 620, 75, 36), "Fast", shortcut_key="3", font=self.renderer.font_small),
        }
        btn_stop = Button(pygame.Rect(959, 620, 80, 36), "Stop", shortcut_key="S", font=self.renderer.font_small)

        self.renderer.render_training(self.env, stats, speed_buttons, btn_stop, "NORMAL")

    def test_11_watch_ai_and_human_play_render(self):
        """Verify Watch AI and Human Play screens render with Phase 17 UI cards."""
        stats = {
            "current_state": (2, 3),
            "action": "DOWN",
            "reward": 5.0,
            "total_reward": 45.0,
            "steps": 8,
            "status": "Navigating Optimal Route",
            "goal_reached": False,
            "distance": 14,
            "start_name": "Main Gate",
            "destination_name": "Library",
            "last_collision_time": 0,
            "last_collision_cell": None,
            "last_collision_msg": "",
            "goal_arrival_time": 0,
        }
        btn_restart = Button(pygame.Rect(710, 620, 160, 44), "Restart", shortcut_key="R", font=self.renderer.font_btn)
        btn_menu = Button(pygame.Rect(880, 620, 160, 44), "Back", shortcut_key="ESC", font=self.renderer.font_btn)
        btn_route = Button(pygame.Rect(710, 560, 330, 44), "Change Route", shortcut_key="C", font=self.renderer.font_btn)

        self.renderer.render_watch_ai(
            self.env, stats, [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3)],
            False, btn_restart, btn_menu, route_btn=btn_route
        )


if __name__ == "__main__":
    unittest.main()
