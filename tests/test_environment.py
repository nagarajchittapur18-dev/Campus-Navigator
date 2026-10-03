"""
Campus Navigator: Phase 19 Environment Unit Test Suite
======================================================
Module: tests/test_environment.py

Verifies:
- reset() initializes coordinates, step counter, and state.
- Valid movement updates agent position across all 4 cardinal directions.
- Invalid movement keeps agent in bounds and applies penalties.
- Goal detection identifies arrival at the target landmark.
- Obstacle detection prevents walking through campus buildings.
- Reward calculation accurately computes step, goal, distance, and zone rewards.
- Terminal state correctly triggers on goal arrival and max steps exhaustion.
"""

import os
import sys
import unittest
from typing import Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from environment.campus_env import CampusEnvironment


class TestCampusEnvironment(unittest.TestCase):
    """Test suite covering all CampusEnvironment specifications."""

    def setUp(self):
        """Create fresh environment instance before each test."""
        self.env = CampusEnvironment(
            rows=config.GRID_ROWS,
            cols=config.GRID_COLS,
            start_pos=config.START_POSITION,
            goal_pos=config.GOAL_POSITION,
            max_steps=50,
        )

    # --------------------------------------------------------------------------
    # 1. reset() Tests
    # --------------------------------------------------------------------------
    def test_reset_initializes_state(self):
        """Verify reset() restores starting coordinates and step counter."""
        # Mutate environment state
        self.env.agent_pos = (5, 5)
        self.env.step_count = 25

        state = self.env.reset()

        self.assertEqual(state, config.START_POSITION)
        self.assertEqual(self.env.agent_pos, config.START_POSITION)
        self.assertEqual(self.env.step_count, 0)
        self.assertFalse(self.env.is_terminal())

    def test_reset_custom_start_and_goal(self):
        """Verify reset() accepts custom start and goal coordinates."""
        custom_start = (2, 3)
        custom_goal = (7, 8)

        state = self.env.reset(start_pos=custom_start, goal_pos=custom_goal)

        self.assertEqual(state, custom_start)
        self.assertEqual(self.env.start_pos, custom_start)
        self.assertEqual(self.env.goal_pos, custom_goal)

    # --------------------------------------------------------------------------
    # 2. Valid Movement Tests
    # --------------------------------------------------------------------------
    def test_valid_movement_cardinal_directions(self):
        """Verify agent moves accurately UP, DOWN, LEFT, RIGHT."""
        # Place agent at open hallway location (5, 5)
        self.env.agent_pos = (5, 5)

        # Move UP (action 0) -> (4, 5)
        next_state, reward, done, info = self.env.step(config.ACTION_UP)
        self.assertEqual(next_state, (4, 5))
        self.assertEqual(self.env.agent_pos, (4, 5))
        self.assertTrue(info["is_valid"])

        # Move DOWN (action 1) -> back to (5, 5)
        next_state, reward, done, info = self.env.step(config.ACTION_DOWN)
        self.assertEqual(next_state, (5, 5))

        # Move LEFT (action 2) -> (5, 4)
        next_state, reward, done, info = self.env.step(config.ACTION_LEFT)
        self.assertEqual(next_state, (5, 4))

        # Move RIGHT (action 3) -> back to (5, 5)
        next_state, reward, done, info = self.env.step(config.ACTION_RIGHT)
        self.assertEqual(next_state, (5, 5))

    # --------------------------------------------------------------------------
    # 3. Invalid Movement Tests
    # --------------------------------------------------------------------------
    def test_invalid_movement_grid_boundaries(self):
        """Verify boundary violations keep agent in place and penalize."""
        # Top-left corner (0, 0)
        self.env.agent_pos = (0, 0)

        # Moving UP from top row is out of bounds
        next_state, reward, done, info = self.env.step(config.ACTION_UP)
        self.assertEqual(next_state, (0, 0))
        self.assertFalse(info["is_valid"])
        self.assertEqual(reward, config.REWARD_INVALID_MOVE)

        # Moving LEFT from column 0 is out of bounds
        next_state, reward, done, info = self.env.step(config.ACTION_LEFT)
        self.assertEqual(next_state, (0, 0))
        self.assertFalse(info["is_valid"])
        self.assertEqual(reward, config.REWARD_INVALID_MOVE)

        # Bottom-right corner (9, 9)
        self.env.agent_pos = (9, 9)
        # Moving DOWN from bottom row
        next_state, reward, done, info = self.env.step(config.ACTION_DOWN)
        self.assertEqual(next_state, (9, 9))
        self.assertFalse(info["is_valid"])

        # Moving RIGHT from rightmost column
        next_state, reward, done, info = self.env.step(config.ACTION_RIGHT)
        self.assertEqual(next_state, (9, 9))
        self.assertFalse(info["is_valid"])

    # --------------------------------------------------------------------------
    # 4. Obstacle Detection Tests
    # --------------------------------------------------------------------------
    def test_obstacle_detection_blocks_movement(self):
        """Verify walking into an obstacle is blocked and penalized."""
        # Obstacle at (0, 4); place agent at (0, 3) adjacent
        self.assertIn((0, 4), self.env.obstacles)
        self.env.agent_pos = (0, 3)

        # Moving RIGHT directly into obstacle at (0, 4)
        next_state, reward, done, info = self.env.step(config.ACTION_RIGHT)

        self.assertEqual(next_state, (0, 3), "Agent walked into obstacle building!")
        self.assertFalse(info["is_valid"])
        self.assertEqual(reward, config.REWARD_INVALID_MOVE)

    def test_get_valid_actions_filters_obstacles(self):
        """Verify get_valid_actions filters out blocked/colliding actions."""
        # At (0, 3), obstacles exist at (0, 4) (RIGHT). UP is out of bounds.
        # Valid moves from (0, 3) are DOWN (1, 3) and LEFT (0, 2).
        self.env.agent_pos = (0, 3)
        valid_actions = self.env.get_valid_actions()

        self.assertNotIn(config.ACTION_UP, valid_actions)     # Out of bounds
        self.assertNotIn(config.ACTION_RIGHT, valid_actions)  # Obstacle at (0, 4)
        self.assertIn(config.ACTION_DOWN, valid_actions)      # Walkable
        self.assertIn(config.ACTION_LEFT, valid_actions)      # Walkable

    # --------------------------------------------------------------------------
    # 5. Goal Detection Tests
    # --------------------------------------------------------------------------
    def test_goal_detection_identifies_arrival(self):
        """Verify stepping into the goal sets goal_reached and terminal flags."""
        # Place agent 1 step above Central Library (9, 9)
        self.env.agent_pos = (8, 9)

        # Move DOWN into Goal (9, 9)
        next_state, reward, done, info = self.env.step(config.ACTION_DOWN)

        self.assertEqual(next_state, config.GOAL_POSITION)
        self.assertTrue(done, "Episode should terminate upon reaching goal!")
        self.assertTrue(info["goal_reached"])
        self.assertEqual(reward, config.REWARD_GOAL)

    # --------------------------------------------------------------------------
    # 6. Reward Calculation Tests
    # --------------------------------------------------------------------------
    def test_reward_calculation_shaping_and_zones(self):
        """Verify scalar rewards for goal, distance shaping, and special zones."""
        # Goal reward
        r_goal = self.env.calculate_reward((8, 9), (9, 9), is_valid=True)
        self.assertEqual(r_goal, config.REWARD_GOAL)

        # Move closer (distance decreases)
        r_closer = self.env.calculate_reward((0, 0), (1, 0), is_valid=True)
        self.assertEqual(r_closer, config.REWARD_MOVE_CLOSER)

        # Move farther (distance increases)
        r_farther = self.env.calculate_reward((1, 0), (0, 0), is_valid=True)
        self.assertEqual(r_farther, config.REWARD_MOVE_FARTHER)

        # Invalid move / collision
        r_invalid = self.env.calculate_reward((0, 0), (0, 0), is_valid=False)
        self.assertEqual(r_invalid, config.REWARD_INVALID_MOVE)

        # Safe Zone reward modifier
        safe_cell = list(self.env.safe_zones)[0]
        r_safe = self.env.calculate_reward((safe_cell[0] - 1, safe_cell[1]), safe_cell, is_valid=True)
        self.assertEqual(r_safe, config.REWARD_MOVE_CLOSER + self.env.reward_safe_zone)

        # Crowded Zone penalty modifier
        crowd_cell = list(self.env.crowded_zones)[0]
        r_crowd = self.env.calculate_reward((crowd_cell[0] - 1, crowd_cell[1]), crowd_cell, is_valid=True)
        self.assertEqual(r_crowd, config.REWARD_MOVE_CLOSER + self.env.reward_crowded_zone)

        # Unsafe Hazard Zone penalty
        unsafe_cell = list(self.env.unsafe_zones)[0]
        r_unsafe = self.env.calculate_reward((unsafe_cell[0] - 1, unsafe_cell[1]), unsafe_cell, is_valid=True)
        self.assertEqual(r_unsafe, self.env.reward_unsafe_zone)

    # --------------------------------------------------------------------------
    # 7. Terminal State Tests
    # --------------------------------------------------------------------------
    def test_terminal_state_on_max_steps_exhaustion(self):
        """Verify episode terminates and penalizes when max_steps horizon is exhausted."""
        self.env.max_steps = 5
        self.env.reset()

        for step in range(4):
            _, _, done, _ = self.env.step(config.ACTION_DOWN)
            self.assertFalse(done)

        # 5th step hits max_steps
        _, reward, done, info = self.env.step(config.ACTION_DOWN)
        self.assertTrue(done, "Episode should terminate when max steps reached!")
        self.assertFalse(info["goal_reached"])
        self.assertEqual(self.env.step_count, 5)


if __name__ == "__main__":
    unittest.main()
