"""
Campus Navigator: Phase 16 Multiple Destinations & Start Locations Test Suite
=============================================================================
Module: tests/test_multi_destinations.py

Verifies:
1. Environment predefined destination & start location configuration and resets.
2. Goal-Conditioned Q-learning agent with (5, 100, 4) tensor slice view and (100, 4) backward compatibility.
3. Greedy autonomous pathfinding verification across all 5 campus destinations.
4. Headless GUI Route Selection Modal opening, button interaction, and state synchronization.
"""

import os
import sys
import unittest
import numpy as np

# Ensure headless Pygame testing before importing pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config
from environment.campus_env import CampusEnvironment
from agent.q_learning import QLearningAgent
from game.game import CampusNavigatorGame


class TestPhase16MultipleDestinations(unittest.TestCase):
    """Test suite for Phase 16 Multiple Destinations and Start Locations."""

    @classmethod
    def setUpClass(cls):
        pygame.init()

    def setUp(self):
        self.env = CampusEnvironment(rows=10, cols=10)
        self.agent = QLearningAgent(rows=10, cols=10, num_actions=4)

    def test_01_environment_landmarks_and_resets(self):
        """Verify environment sets destination and start coordinates properly."""
        # Check all 5 destinations exist and match coordinates
        expected_dests = {
            config.DEST_LIBRARY: (9, 9),
            config.DEST_MAIN_BLOCK: (1, 8),
            config.DEST_COMPUTER_LAB: (4, 3),
            config.DEST_CAFETERIA: (3, 2),
            config.DEST_SPORTS_GROUND: (9, 0),
        }
        for name, coord in expected_dests.items():
            self.assertIn(name, self.env.destinations)
            self.assertEqual(self.env.destinations[name], coord)
            
            # Test setting destination by string
            res_coord = self.env.set_destination(name)
            self.assertEqual(res_coord, coord)
            self.assertEqual(self.env.goal_pos, coord)
            self.assertEqual(self.env.current_destination_name, name)

        # Check all 5 start locations
        expected_starts = {
            config.START_MAIN_GATE: (0, 0),
            config.START_NORTH_GATE: (0, 9),
            config.START_WEST_GATE: (9, 1),
            config.START_HOSTEL_PLAZA: (7, 4),
            config.START_CAFETERIA: (3, 2),
        }
        for name, coord in expected_starts.items():
            self.assertIn(name, self.env.start_locations)
            self.assertEqual(self.env.start_locations[name], coord)

            # Test setting start by string
            res_coord = self.env.set_start(name)
            self.assertEqual(res_coord, coord)
            self.assertEqual(self.env.start_pos, coord)
            self.assertEqual(self.env.current_start_name, name)

            # Resetting environment should place agent at start_pos
            agent_pos = self.env.reset()
            self.assertEqual(agent_pos, coord)
            self.assertEqual(self.env.agent_pos, coord)

    def test_02_goal_conditioned_agent_tensor_and_slice(self):
        """Verify Goal-Conditioned Q-learning agent uses (5, 100, 4) tensor with 2D slice view."""
        self.assertEqual(self.agent.multi_q_table.shape, (5, 100, 4))
        self.assertEqual(self.agent.q_table.shape, (100, 4))
        self.assertEqual(self.agent.num_destinations, 5)

        # Test switching destinations updates destination ID and slice view
        for dest_id, (name, coord) in enumerate(config.DESTINATIONS.items()):
            active_id = self.agent.set_destination(name)
            self.assertEqual(active_id, dest_id)
            self.assertEqual(self.agent.current_dest_id, dest_id)
            self.assertEqual(self.agent.current_dest_name, name)

            # In-place mutation test: updating q_table must mutate multi_q_table directly
            test_val = 42.5 + dest_id
            self.agent.q_table[15, 2] = test_val
            self.assertEqual(self.agent.multi_q_table[dest_id, 15, 2], test_val)

        # Reset agent and verify all zeros
        self.agent.reset()
        self.assertEqual(np.count_nonzero(self.agent.multi_q_table), 0)
        self.assertEqual(self.agent.q_table.shape, (100, 4))

    def test_03_autonomous_policy_navigation_all_5_destinations(self):
        """Verify trained agent navigates successfully to all 5 campus destinations."""
        multi_path = getattr(config, "Q_TABLE_MULTI_SAVE_PATH", os.path.join(config.MODELS_DIR, "q_table_multi.npy"))
        self.assertTrue(os.path.exists(multi_path), f"Multi-destination model not found at {multi_path}")

        loaded = self.agent.load_model(config.Q_TABLE_SAVE_PATH)
        self.assertTrue(loaded)
        self.agent.epsilon = 0.0

        destinations = list(config.DESTINATIONS.keys())
        self.assertEqual(len(destinations), 5)

        for dest_name in destinations:
            goal_coord = config.DESTINATIONS[dest_name]
            self.env.set_start(config.START_MAIN_GATE)
            self.env.set_destination(dest_name)
            self.agent.set_destination(dest_name)

            state = self.env.reset()
            done = False
            steps = 0
            max_steps = 50

            while not done and steps < max_steps:
                action = self.agent.get_best_action(state)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                steps += 1

            self.assertTrue(
                info.get("goal_reached", False),
                f"Agent failed to reach destination '{dest_name}' ({goal_coord}) within {max_steps} steps. Ended at {state}."
            )
            self.assertEqual(state, goal_coord)

    def test_04_route_selection_modal_and_game_integration(self):
        """Verify GUI Route Selection Modal initialization, interaction, and state synchronization."""
        game = CampusNavigatorGame()

        # Initial state should be baseline
        self.assertEqual(game.selected_start_name, config.START_MAIN_GATE)
        self.assertEqual(game.selected_dest_name, config.DEST_LIBRARY)
        self.assertFalse(game.show_route_modal)

        # Open route modal
        game._open_route_modal()
        self.assertTrue(game.show_route_modal)
        self.assertEqual(game.temp_selected_start, config.START_MAIN_GATE)
        self.assertEqual(game.temp_selected_dest, config.DEST_LIBRARY)

        # Check modal buttons count
        self.assertEqual(len(game.modal_start_buttons), 5)
        self.assertEqual(len(game.modal_dest_buttons), 5)

        # Simulate user selecting "West Gate" and "Computer Lab"
        game.temp_selected_start = config.START_WEST_GATE
        game.temp_selected_dest = config.DEST_COMPUTER_LAB

        # Apply selection
        game._apply_route_selection()
        self.assertFalse(game.show_route_modal)
        self.assertEqual(game.selected_start_name, config.START_WEST_GATE)
        self.assertEqual(game.selected_dest_name, config.DEST_COMPUTER_LAB)

        # Verify environment and agent synchronization
        self.assertEqual(game.env.start_pos, config.START_LOCATIONS[config.START_WEST_GATE])
        self.assertEqual(game.env.goal_pos, config.DESTINATIONS[config.DEST_COMPUTER_LAB])
        self.assertEqual(game.agent.current_dest_id, 2)  # Computer Lab is ID 2

        # Verify agent resets to selected start position
        agent_pos = game.env.reset()
        self.assertEqual(agent_pos, config.START_LOCATIONS[config.START_WEST_GATE])


if __name__ == "__main__":
    unittest.main()
