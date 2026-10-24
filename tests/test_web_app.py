"""
Campus Navigator: Web Interface Unit & Integration Tests
=========================================================
Module: tests/test_web_app.py

Validates the Gradio web application interface, session-isolated state,
touch controls, autonomous AI demonstration, Q-table inspector, headless
training, results loader, and error resilience without opening a browser.
"""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from web.app import (
    render_campus_svg,
    render_telemetry_html,
    make_initial_human_state,
    handle_human_move,
    handle_human_reset,
    make_initial_ai_state,
    handle_ai_step,
    handle_ai_run_full,
    handle_ai_reset,
    render_q_table_inspector,
    execute_web_training,
    load_project_results_overview,
    build_app,
)


class TestWebApp(unittest.TestCase):
    """Test suite for the Gradio web application."""

    def test_01_svg_rendering(self):
        """Verify SVG grid is generated cleanly with proper viewBox and elements."""
        svg = render_campus_svg(
            agent_pos=(0, 0),
            start_pos=(0, 0),
            goal_pos=(9, 9),
            obstacles=config.DEFAULT_OBSTACLES,
            mode="human",
            dest_name="Library",
        )
        self.assertIsInstance(svg, str)
        self.assertTrue(svg.startswith("<svg"))
        self.assertTrue(svg.endswith("</svg>"))
        self.assertIn('viewBox="0 0', svg)
        self.assertIn("GOAL", svg)
        self.assertIn("GATE", svg)

    def test_02_telemetry_rendering(self):
        """Verify live telemetry HTML card generation."""
        telemetry = render_telemetry_html(
            agent_pos=(2, 3),
            current_action="UP",
            total_reward=15.0,
            last_reward=-1.0,
            steps=5,
            collisions=0,
            status_msg="Moving forward",
            status_type="info",
            dest_name="Library",
        )
        self.assertIsInstance(telemetry, str)
        self.assertIn("(2, 3)", telemetry)
        self.assertIn("Index 23", telemetry)
        self.assertIn("+15.0", telemetry)
        self.assertIn("-1.0", telemetry)

    def test_03_human_state_and_movement(self):
        """Verify session-isolated human navigation state and moves."""
        state = make_initial_human_state("Library")
        self.assertEqual(state["agent_pos"], (0, 0))
        self.assertEqual(state["steps"], 0)
        self.assertEqual(state["total_reward"], 0.0)
        self.assertFalse(state["done"])

        # Move DOWN (Row 0 -> Row 1)
        state, svg, tele = handle_human_move(config.ACTION_DOWN, state)
        self.assertEqual(state["agent_pos"], (1, 0))
        self.assertEqual(state["steps"], 1)
        self.assertIn("(1, 0)", tele)

        # Move LEFT into left wall boundary -> Collision detected
        state, svg, tele = handle_human_move(config.ACTION_LEFT, state)
        self.assertEqual(state["collisions"], 1)
        self.assertEqual(state["agent_pos"], (1, 0))  # Remains in place
        self.assertIn("Collision", state["status_msg"])

        # Reset
        state, svg, tele = handle_human_reset("Cafeteria", state)
        self.assertEqual(state["agent_pos"], (0, 0))
        self.assertEqual(state["steps"], 0)
        self.assertEqual(state["dest_name"], "Cafeteria")

    def test_04_ai_step_and_full_run(self):
        """Verify trained AI autonomous demonstration and optimal navigation."""
        state = make_initial_ai_state("Library")
        self.assertTrue(state["model_loaded"])

        # Take 1 step
        state, svg, tele, q_table_html = handle_ai_step(state)
        self.assertEqual(state["steps"], 1)
        self.assertIn("Q-Values Inspector", q_table_html)

        # Run full route to Library
        full_state, svg, tele, q_table_html = handle_ai_run_full("Library", state)
        self.assertTrue(full_state["done"])
        self.assertTrue(full_state["success"])
        self.assertEqual(full_state["agent_pos"], (9, 9))
        self.assertEqual(full_state["steps"], 18)
        self.assertEqual(full_state["collisions"], 0)

    def test_05_q_table_inspector(self):
        """Verify Q-values inspector renders correct values and identifies best action."""
        state = make_initial_ai_state("Library")
        inspector_html = render_q_table_inspector(state["agent"], (0, 0))
        self.assertIn("BEST", inspector_html)
        self.assertIn("UP", inspector_html)
        self.assertIn("DOWN", inspector_html)

    def test_06_web_training_execution(self):
        """Verify headless web training produces real metrics and charts."""
        summary_html, fig, export_path = execute_web_training(
            episodes=60,
            alpha=0.1,
            gamma=0.9,
            eps_decay=0.99,
        )
        self.assertIn("Training Successfully Completed", summary_html)
        self.assertIsNotNone(fig)
        self.assertTrue(os.path.isfile(export_path))
        # Verify saved table is valid numpy array
        arr = np.load(export_path)
        self.assertEqual(arr.shape, (100, 4))

    def test_07_results_overview_loader(self):
        """Verify results overview loader returns actual KPI cards and figures."""
        kpi_html, df, plots = load_project_results_overview()
        self.assertIn("SUCCESS RATE", kpi_html.upper())
        self.assertIn("AVERAGE STEPS", kpi_html.upper())
        self.assertGreater(len(plots), 0)

    def test_08_missing_model_resilience(self):
        """Verify application handles missing model gracefully without crashing."""
        fake_state = {
            "model_loaded": False,
            "agent_pos": (0, 0),
            "dest_name": "Library",
        }
        state, svg, tele, q_out = handle_ai_step(fake_state)
        self.assertIn("Trained Q-table/model not found", tele)
        self.assertIn("Trained Q-table/model not found", q_out)

    def test_09_build_gradio_app(self):
        """Verify Gradio Blocks application compiles with all tabs and components."""
        demo = build_app()
        self.assertIsNotNone(demo)
        self.assertEqual(demo.title, "Campus Navigator - Q-Learning RL")


if __name__ == "__main__":
    unittest.main()
