"""
Campus Navigator: Virtual Campus Environment (Phase 1)
======================================================
Module: environment/campus_env.py

RL Concept - ENVIRONMENT (Markov Decision Process):
In Reinforcement Learning, the Environment models the world the Agent navigates.
It formalizes the problem as a Markov Decision Process (MDP):
    - State Space (S): Discrete (row, column) coordinates on a 10x10 grid.
    - Action Space (A): 4 discrete moves (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT).
    - Transition Dynamics P(S' | S, A): Deterministic movement to adjacent cells
      with collision detection for walls and campus obstacles.
    - Reward Function R(S, A, S'): Immediate scalar feedback guiding the agent
      toward the destination while penalizing collisions and hazard zones.
    - Terminal States: Reaching the Central Library (Goal) or exceeding max steps.

NO PYGAME DEPENDENCIES:
This module is strictly decoupled from any GUI or game engine. It runs
completely standalone from the command line or within test scripts.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

# Ensure project root is accessible when script is run standalone
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config


class CampusEnvironment:
    """
    10x10 Gridworld Environment representing a University Campus.

    Cell Types:
    - WALKABLE ('.'): Open corridors and pathways.
    - OBSTACLE ('#'): Impassable buildings and construction sites.
    - START    ('S'): Campus Main Gate at (0, 0).
    - GOAL     ('G'): Central Library at (9, 9).
    - SAFE_ZONE   ('Z'): Student support lounges / well-lit plazas (small positive reward).
    - CROWDED_ZONE ('C'): High-traffic congested pedestrian corridors (small negative reward).
    - UNSAFE_ZONE ('!'): Hazardous / slippery construction corridors (larger negative reward).
    """

    def __init__(
        self,
        rows: int = config.GRID_ROWS,
        cols: int = config.GRID_COLS,
        start_pos: Tuple[int, int] = config.START_POSITION,
        goal_pos: Tuple[int, int] = config.GOAL_POSITION,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        safe_zones: Optional[List[Tuple[int, int]]] = None,
        unsafe_zones: Optional[List[Tuple[int, int]]] = None,
        max_steps: int = config.MAX_STEPS_PER_EPISODE,
        crowded_zones: Optional[List[Tuple[int, int]]] = None,
        reward_safe_zone: Optional[float] = None,
        reward_crowded_zone: Optional[float] = None,
        reward_unsafe_zone: Optional[float] = None,
        reward_goal: Optional[float] = None,
        reward_normal_step: Optional[float] = None,
        reward_move_closer: Optional[float] = None,
        reward_move_farther: Optional[float] = None,
        reward_invalid_move: Optional[float] = None,
        reward_max_steps_exceeded: Optional[float] = None,
        use_distance_shaping: Optional[bool] = None,
        reward_profile: Optional[Any] = None,
    ) -> None:
        """
        Initialize the deterministic 10x10 Campus Environment.

        Args:
            rows: Total number of rows in the campus grid (default: 10).
            cols: Total number of columns in the campus grid (default: 10).
            start_pos: Starting coordinate (row, col) (default: (0, 0)).
            goal_pos: Target destination coordinate (row, col) (default: (9, 9)).
            obstacles: List of impassable obstacle coordinates.
            safe_zones: Optional list of safe zone coordinates.
            unsafe_zones: Optional list of unsafe/hazard zone coordinates.
            max_steps: Maximum steps allowed per episode before truncation.
            crowded_zones: Optional list of congested pedestrian corridor coordinates.
            reward_safe_zone: Configurable reward bonus for entering safe zones.
            reward_crowded_zone: Configurable penalty for navigating crowded zones.
            reward_unsafe_zone: Configurable penalty for traversing hazard zones.
            reward_goal: Reward for reaching destination (default: +100).
            reward_normal_step: Baseline cost per movement step (default: -1).
            reward_move_closer: Shaping reward when reducing distance (default: +5).
            reward_move_farther: Shaping penalty when increasing distance (default: -3).
            reward_invalid_move: Collision penalty for wall/obstacle (default: -10).
            reward_max_steps_exceeded: Penalty when exceeding max episode steps (default: -20).
            use_distance_shaping: Whether to apply distance-based closer/farther reward shaping.
            reward_profile: Optional RewardProfile instance (Phase 21).
        """
        self.rows: int = rows
        self.cols: int = cols
        self.start_pos: Tuple[int, int] = start_pos
        self.goal_pos: Tuple[int, int] = goal_pos
        self.max_steps: int = max_steps

        # Deterministic layout initialization
        self.obstacles: Set[Tuple[int, int]] = set(
            obstacles if obstacles is not None else config.DEFAULT_OBSTACLES
        )
        self.safe_zones: Set[Tuple[int, int]] = set(
            safe_zones if safe_zones is not None else config.DEFAULT_SAFE_ZONES
        )
        self.crowded_zones: Set[Tuple[int, int]] = set(
            crowded_zones if crowded_zones is not None else config.DEFAULT_CROWDED_ZONES
        )
        self.unsafe_zones: Set[Tuple[int, int]] = set(
            unsafe_zones if unsafe_zones is not None else config.DEFAULT_UNSAFE_ZONES
        )

        # Configurable special cell reward parameters (Phase 15)
        self.reward_safe_zone: float = (
            float(reward_safe_zone) if reward_safe_zone is not None else float(config.REWARD_SAFE_ZONE)
        )
        self.reward_crowded_zone: float = (
            float(reward_crowded_zone) if reward_crowded_zone is not None else float(config.REWARD_CROWDED_ZONE)
        )
        self.reward_unsafe_zone: float = (
            float(reward_unsafe_zone) if reward_unsafe_zone is not None else float(config.REWARD_UNSAFE_ZONE)
        )

        # Configurable global reward parameters (Phase 21: Configurable Reward Profiles)
        self.reward_goal: float = (
            float(reward_goal) if reward_goal is not None else float(config.REWARD_GOAL)
        )
        self.reward_normal_step: float = (
            float(reward_normal_step) if reward_normal_step is not None else float(config.REWARD_NORMAL_STEP)
        )
        self.reward_move_closer: float = (
            float(reward_move_closer) if reward_move_closer is not None else float(config.REWARD_MOVE_CLOSER)
        )
        self.reward_move_farther: float = (
            float(reward_move_farther) if reward_move_farther is not None else float(config.REWARD_MOVE_FARTHER)
        )
        self.reward_invalid_move: float = (
            float(reward_invalid_move) if reward_invalid_move is not None else float(config.REWARD_INVALID_MOVE)
        )
        self.reward_max_steps_exceeded: float = (
            float(reward_max_steps_exceeded) if reward_max_steps_exceeded is not None else float(config.REWARD_MAX_STEPS_EXCEEDED)
        )
        self.use_distance_shaping: bool = True if use_distance_shaping is None else bool(use_distance_shaping)
        self.active_reward_profile_name: str = "Default"

        if reward_profile is not None:
            self.set_reward_profile(reward_profile)

        # Dynamic state tracking
        self.agent_pos: Tuple[int, int] = self.start_pos
        self.step_count: int = 0

        # Phase 16: Predefined Multiple Destinations & Start Locations
        self.destinations: Dict[str, Tuple[int, int]] = dict(config.DESTINATIONS)
        self.start_locations: Dict[str, Tuple[int, int]] = dict(config.START_LOCATIONS)
        self.current_destination_name: str = config.DEST_LIBRARY
        self.current_start_name: str = config.START_MAIN_GATE

    # --------------------------------------------------------------------------
    # Core Environment Interface Methods
    # --------------------------------------------------------------------------

    def set_destination(self, goal: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
        """
        Configure the target destination landmark (Phase 16).

        Args:
            goal: Landmark name string (e.g. 'Sports Ground') or coordinate tuple (r, c).

        Returns:
            goal_pos: Updated target coordinate tuple.
        """
        if isinstance(goal, str):
            if goal in self.destinations:
                self.goal_pos = self.destinations[goal]
                self.current_destination_name = goal
            else:
                raise ValueError(
                    f"Unknown destination landmark '{goal}'. Valid options: {list(self.destinations.keys())}"
                )
        elif isinstance(goal, (tuple, list)):
            if len(goal) < 2:
                raise ValueError(f"Goal coordinate {goal} must contain 2 elements (row, col).")
            gr, gc = int(goal[0]), int(goal[1])
            if not (0 <= gr < self.rows and 0 <= gc < self.cols):
                err = f"Destination coordinate ({gr}, {gc}) is out of campus bounds ({self.rows}x{self.cols})."
                try:
                    from utils.logger import log_error
                    log_error(err)
                except ImportError:
                    pass
                raise ValueError(err)
            self.goal_pos = (gr, gc)
            found = False
            for name, coord in self.destinations.items():
                if coord == self.goal_pos:
                    self.current_destination_name = name
                    found = True
                    break
            if not found:
                self.current_destination_name = f"Custom ({self.goal_pos[0]}, {self.goal_pos[1]})"
        else:
            raise TypeError(f"Destination must be a string or coordinate tuple, got {type(goal)}.")
        return self.goal_pos

    def set_start(self, start: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
        """
        Configure the starting location landmark (Phase 16).

        Args:
            start: Landmark name string (e.g. 'Hostel Plaza') or coordinate tuple (r, c).

        Returns:
            start_pos: Updated starting coordinate tuple.
        """
        if isinstance(start, str):
            if start in self.start_locations:
                self.start_pos = self.start_locations[start]
                self.current_start_name = start
            else:
                raise ValueError(
                    f"Unknown start landmark '{start}'. Valid options: {list(self.start_locations.keys())}"
                )
        elif isinstance(start, (tuple, list)):
            if len(start) < 2:
                raise ValueError(f"Start coordinate {start} must contain 2 elements (row, col).")
            sr, sc = int(start[0]), int(start[1])
            if not (0 <= sr < self.rows and 0 <= sc < self.cols):
                err = f"Start coordinate ({sr}, {sc}) is out of campus bounds ({self.rows}x{self.cols})."
                try:
                    from utils.logger import log_error
                    log_error(err)
                except ImportError:
                    pass
                raise ValueError(err)
            self.start_pos = (sr, sc)
            found = False
            for name, coord in self.start_locations.items():
                if coord == self.start_pos:
                    self.current_start_name = name
                    found = True
                    break
            if not found:
                self.current_start_name = f"Custom ({self.start_pos[0]}, {self.start_pos[1]})"
        else:
            raise TypeError(f"Start location must be a string or coordinate tuple, got {type(start)}.")
        return self.start_pos

    def set_reward_profile(self, profile: Any) -> None:
        """
        Configure the environment's reward function using a RewardProfile (Phase 21).

        Args:
            profile: RewardProfile instance or dictionary.
        """
        if hasattr(profile, "to_dict"):
            p = profile.to_dict()
        elif isinstance(profile, dict):
            p = profile
        else:
            p = {
                "name": getattr(profile, "name", "Custom"),
                "reward_goal": getattr(profile, "reward_goal", self.reward_goal),
                "reward_step": getattr(profile, "reward_step", self.reward_normal_step),
                "reward_obstacle": getattr(profile, "reward_obstacle", self.reward_invalid_move),
                "reward_closer": getattr(profile, "reward_closer", self.reward_move_closer),
                "reward_farther": getattr(profile, "reward_farther", self.reward_move_farther),
                "reward_unsafe": getattr(profile, "reward_unsafe", self.reward_unsafe_zone),
                "reward_safe": getattr(profile, "reward_safe", self.reward_safe_zone),
                "reward_crowded": getattr(profile, "reward_crowded", self.reward_crowded_zone),
                "reward_max_steps": getattr(profile, "reward_max_steps", self.reward_max_steps_exceeded),
                "use_distance_shaping": getattr(profile, "use_distance_shaping", self.use_distance_shaping),
            }

        self.active_reward_profile_name = str(p.get("name", "Custom"))
        self.reward_goal = float(p.get("reward_goal", self.reward_goal))
        self.reward_normal_step = float(p.get("reward_step", p.get("reward_normal_step", self.reward_normal_step)))
        self.reward_invalid_move = float(p.get("reward_obstacle", p.get("reward_invalid_move", self.reward_invalid_move)))

        closer_val = p.get("reward_closer")
        farther_val = p.get("reward_farther")
        self.reward_move_closer = float(closer_val) if closer_val is not None else self.reward_normal_step
        self.reward_move_farther = float(farther_val) if farther_val is not None else self.reward_normal_step

        self.reward_unsafe_zone = float(p.get("reward_unsafe", p.get("reward_unsafe_zone", self.reward_unsafe_zone)))
        self.reward_safe_zone = float(p.get("reward_safe", p.get("reward_safe_zone", self.reward_safe_zone)))
        self.reward_crowded_zone = float(p.get("reward_crowded", p.get("reward_crowded_zone", self.reward_crowded_zone)))
        self.reward_max_steps_exceeded = float(p.get("reward_max_steps", p.get("reward_max_steps_exceeded", self.reward_max_steps_exceeded)))
        self.use_distance_shaping = bool(p.get("use_distance_shaping", self.use_distance_shaping))

    def reset(
        self,
        start_pos: Optional[Union[str, Tuple[int, int]]] = None,
        goal_pos: Optional[Union[str, Tuple[int, int]]] = None,
    ) -> Tuple[int, int]:
        """
        Reset the environment for a new episode.

        Optionally accepts new start and goal coordinates or landmark names (Phase 16).

        Returns:
            initial_state: Starting coordinate (row, col).
        """
        if start_pos is not None:
            self.set_start(start_pos)
        if goal_pos is not None:
            self.set_destination(goal_pos)

        self.agent_pos = self.start_pos
        self.step_count = 0
        return self.agent_pos

    def get_state(self) -> Tuple[int, int]:
        """
        Return the current state of the agent.

        RL Concept - STATE REPRESENTATION:
        The state is completely observed and represented as (row, column).

        Returns:
            state: Tuple (row, col) where row in [0, rows-1], col in [0, cols-1].
        """
        return self.agent_pos

    def get_distance_to_goal(self, state: Optional[Tuple[int, int]] = None) -> int:
        """
        Calculate the Manhattan distance from a state to the Goal.

        Manhattan Distance Formula:
            D(state, goal) = |row - goal_row| + |col - goal_col|

        Args:
            state: Optional state coordinate. Defaults to current agent_pos.

        Returns:
            distance: Integer Manhattan distance in grid units.
        """
        curr = state if state is not None else self.agent_pos
        return abs(curr[0] - self.goal_pos[0]) + abs(curr[1] - self.goal_pos[1])

    def is_valid_move(self, action: int, state: Optional[Tuple[int, int]] = None) -> bool:
        """
        Check whether an action from a given state is valid and passable.

        A move is valid if and only if:
        1. It does not go out of the campus grid boundaries.
        2. The target cell is not an obstacle/building.

        Args:
            action: Integer action (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT).
            state: Optional starting state. Defaults to current agent_pos.

        Returns:
            is_valid: True if passable, False if hitting wall or obstacle.
        """
        if not isinstance(action, (int, np.integer)) or action not in config.ACTION_DELTA:
            return False

        curr = state if state is not None else self.agent_pos
        delta_r, delta_c = config.ACTION_DELTA[action]
        target_r = curr[0] + delta_r
        target_c = curr[1] + delta_c

        # 1. Boundary check
        if not (0 <= target_r < self.rows and 0 <= target_c < self.cols):
            return False

        # 2. Obstacle collision check
        if (target_r, target_c) in self.obstacles:
            return False

        return True

    def get_valid_actions(self, state: Optional[Tuple[int, int]] = None) -> List[int]:
        """
        Get all valid, non-colliding actions available from the specified state.

        Args:
            state: Optional state coordinate. Defaults to current agent_pos.

        Returns:
            valid_actions: List of valid action integers [0..3].
        """
        curr = state if state is not None else self.agent_pos
        return [act for act in config.ACTIONS if self.is_valid_move(act, curr)]

    def is_terminal(self) -> bool:
        """
        Check if the current episode has concluded.

        Terminal conditions:
        1. Goal reached: Student agent has arrived at the Central Library.
        2. Truncation: Maximum step limit reached without reaching goal.

        Returns:
            is_terminal: True if episode should terminate, False otherwise.
        """
        goal_reached = (self.agent_pos == self.goal_pos)
        steps_exhausted = (self.step_count >= self.max_steps)
        return goal_reached or steps_exhausted

    def calculate_reward(
        self,
        current_state: Tuple[int, int],
        next_state: Tuple[int, int],
        is_valid: bool = True,
        is_max_steps: bool = False,
    ) -> float:
        """
        Calculate the scalar reward for a state transition.

        RL Concept - REWARD FUNCTION:
        Guides agent learning through reward shaping and penalties:
        - Goal reached:               +100.0
        - Move closer to goal:          +5.0
        - Move farther from goal:       -3.0
        - Normal movement (neutral):    -1.0
        - Obstacle / Invalid move:     -10.0
        - Unsafe zone:                 -10.0
        - Maximum steps exceeded:      -20.0

        Args:
            current_state: State (row, col) before taking action.
            next_state: Resulting state (row, col) after transition.
            is_valid: False if move collided with wall or obstacle.
            is_max_steps: True if step limit was exceeded without reaching goal.

        Returns:
            reward: Calculated float reward value.
        """
        # 1. Step limit exceeded penalty
        if is_max_steps and next_state != self.goal_pos:
            return self.reward_max_steps_exceeded

        # 2. Collision penalty (Out of bounds or Obstacle)
        if not is_valid:
            return self.reward_invalid_move

        # 3. Goal reached reward
        if next_state == self.goal_pos:
            return self.reward_goal

        # 4. Unsafe zone penalty (larger negative hazard penalty)
        if next_state in self.unsafe_zones:
            return self.reward_unsafe_zone

        # 5. Distance-based base reward shaping
        if self.use_distance_shaping:
            curr_dist = self.get_distance_to_goal(current_state)
            next_dist = self.get_distance_to_goal(next_state)

            if next_dist < curr_dist:
                base_reward = self.reward_move_closer
            elif next_dist > curr_dist:
                base_reward = self.reward_move_farther
            else:
                base_reward = self.reward_normal_step
        else:
            base_reward = self.reward_normal_step

        # 6. Special zone modifiers (Safe zone: small positive reward; Crowded zone: small negative reward)
        if next_state in self.safe_zones:
            return base_reward + self.reward_safe_zone
        elif next_state in self.crowded_zones:
            return base_reward + self.reward_crowded_zone

        return base_reward

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool, Dict]:
        """
        Execute an action and transition the environment.

        RL Concept - MARKOV TRANSITION STEP:
        Given current state S and chosen action A:
        1. Validates action against boundaries and campus obstacles.
        2. If valid: Agent transitions to candidate state S'.
           If invalid: Agent remains in place (S' = S).
        3. Computes the immediate reward R(S, A, S').
        4. Evaluates if the new state is terminal.
        5. Packages diagnostic info for logging and debugging.

        Args:
            action: Integer action (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT).

        Returns:
            next_state: Updated agent coordinate (row, col).
            reward: Immediate scalar feedback reward.
            done: Boolean flag indicating terminal state.
            info: Diagnostic dictionary (event, step count, distance, etc.).
        """
        self.step_count += 1
        prev_state = self.agent_pos

        # Check for invalid action type or value (Phase 18 Error Handling)
        if not isinstance(action, (int, np.integer)) or action not in config.ACTION_DELTA:
            try:
                from utils.logger import log_warning
                log_warning(
                    f"Invalid action '{action}' (type: {type(action).__name__}) passed to environment step(). "
                    f"Valid actions are integers in [0..{len(config.ACTION_DELTA) - 1}]."
                )
            except ImportError:
                pass
            valid = False
            event = "invalid_action"
            self.agent_pos = prev_state
        else:
            valid = self.is_valid_move(action, prev_state)

            if valid:
                delta_r, delta_c = config.ACTION_DELTA[action]
                self.agent_pos = (prev_state[0] + delta_r, prev_state[1] + delta_c)

                if self.agent_pos == self.goal_pos:
                    event = "goal_reached"
                elif self.agent_pos in self.unsafe_zones:
                    event = "entered_unsafe_zone"
                elif self.agent_pos in self.safe_zones:
                    event = "entered_safe_zone"
                elif self.agent_pos in self.crowded_zones:
                    event = "entered_crowded_zone"
                else:
                    curr_dist = self.get_distance_to_goal(self.agent_pos)
                    prev_dist = self.get_distance_to_goal(prev_state)
                    if curr_dist < prev_dist:
                        event = "moved_closer"
                    elif curr_dist > prev_dist:
                        event = "moved_farther"
                    else:
                        event = "normal_step"
            else:
                # Invalid move (hit wall or obstacle): stay in current position
                self.agent_pos = prev_state
                event = "invalid_move"

        done = self.is_terminal()
        is_max_steps = (self.step_count >= self.max_steps and self.agent_pos != self.goal_pos)
        if is_max_steps:
            event = "max_steps_exceeded"

        reward = self.calculate_reward(
            current_state=prev_state,
            next_state=self.agent_pos,
            is_valid=valid,
            is_max_steps=is_max_steps,
        )

        info = {
            "event": event,
            "step_count": self.step_count,
            "prev_state": prev_state,
            "next_state": self.agent_pos,
            "action": config.ACTIONS.get(action, "UNKNOWN"),
            "distance_to_goal": self.get_distance_to_goal(self.agent_pos),
            "is_valid": valid,
            "goal_reached": (self.agent_pos == self.goal_pos),
            "is_safe_zone": (self.agent_pos in self.safe_zones),
            "is_crowded_zone": (self.agent_pos in self.crowded_zones),
            "is_unsafe_zone": (self.agent_pos in self.unsafe_zones),
        }

        return self.agent_pos, reward, done, info

    def render_ascii(self, show_agent: bool = True) -> str:
        """
        Generate and return a formatted ASCII grid representation of the campus.

        Symbols:
            'S' = Campus Main Gate (Start)
            'G' = Central Library (Goal)
            '#' = Campus Obstacle / Building
            '.' = Walkable Pathway
            'Z' = Safe Zone (Student lounge / well-lit plaza)
            'C' = Crowded Zone (Pedestrian congestion / corridor delay)
            '!' = Unsafe Zone (Hazard / slippery corridor)
            'A' = Student Agent Current Position

        Args:
            show_agent: If True, renders 'A' at current agent position.

        Returns:
            grid_str: Multi-line formatted ASCII string.
        """
        lines = []
        for r in range(self.rows):
            row_chars = []
            for c in range(self.cols):
                pos = (r, c)
                if show_agent and pos == self.agent_pos:
                    row_chars.append("A")
                elif pos == self.start_pos:
                    row_chars.append("S")
                elif pos == self.goal_pos:
                    row_chars.append("G")
                elif pos in self.obstacles:
                    row_chars.append("#")
                elif pos in self.unsafe_zones:
                    row_chars.append("!")
                elif pos in self.safe_zones:
                    row_chars.append("Z")
                elif pos in self.crowded_zones:
                    row_chars.append("C")
                else:
                    row_chars.append(".")
            lines.append(" ".join(row_chars))
        return "\n".join(lines)


# ==============================================================================
# CLI Standalone Verification & Test Suite
# ==============================================================================

def run_environment_demo() -> None:
    """
    Demonstrate all Phase 1 Environment capabilities from the command line:
    1. Resets the environment.
    2. Prints starting state & ASCII campus layout.
    3. Performs several actions (showing closer, farther, normal moves).
    4. Displays next state, reward, and info dictionary.
    5. Demonstrates invalid movement handling (wall boundary & obstacle collision).
    6. Traverses to the goal and verifies goal detection.
    """
    print("=" * 72)
    print(" CAMPUS NAVIGATOR: PHASE 1 ENVIRONMENT DEMO & VERIFICATION")
    print("=" * 72)

    # 1. Initialize & Reset
    env = CampusEnvironment()
    initial_state = env.reset()
    print(f"\n[1] Environment Initialized & Reset:")
    print(f"    - Grid Size     : {env.rows}x{env.cols}")
    print(f"    - Starting State: {initial_state} (Main Gate)")
    print(f"    - Goal State    : {env.goal_pos} (Central Library)")
    print(f"    - Distance to Goal: {env.get_distance_to_goal()} units")

    # 2. Print Starting ASCII Layout
    print("\n[2] Initial Campus ASCII Map (A = Agent, S = Gate, G = Library, # = Obstacle):")
    print("-" * 35)
    print(env.render_ascii(show_agent=True))
    print("-" * 35)

    # 3. Valid Actions Check
    valid_acts = env.get_valid_actions()
    valid_act_names = [config.ACTIONS[a] for a in valid_acts]
    print(f"\n[3] Valid Actions Available from Start {initial_state}:")
    print(f"    - Action indices: {valid_acts}")
    print(f"    - Action names  : {valid_act_names} (UP & LEFT are blocked by campus boundary)")

    # 4. Perform Several Moves
    print("\n[4] Performing Sequence of Step Actions:")
    test_sequence = [
        (config.ACTION_DOWN, "Move DOWN (closer to goal)"),
        (config.ACTION_RIGHT, "Move RIGHT (closer to goal)"),
        (config.ACTION_RIGHT, "Move RIGHT (closer to goal)"),
        (config.ACTION_LEFT, "Move LEFT (farther from goal)"),
    ]

    for action, desc in test_sequence:
        prev = env.get_state()
        next_s, reward, done, info = env.step(action)
        print(
            f"    -> Action: {info['action']:<5} | Desc: {desc:<28} | "
            f"State: {prev} -> {next_s} | Reward: {reward:+5.1f} | Done: {done}"
        )

    # 5. Invalid Movement Handling
    print("\n[5] Testing Invalid Movement Handling:")
    # Reset to start (0, 0)
    env.reset()
    print(f"    Resetting agent to {env.get_state()}")

    # A) Try moving UP out-of-bounds
    next_s, reward, done, info = env.step(config.ACTION_UP)
    print(
        f"    -> Out-of-bounds UP   : State={next_s} | "
        f"Reward={reward:+5.1f} (Penalty) | Event={info['event']} | Valid={info['is_valid']}"
    )

    # B) Try moving LEFT out-of-bounds
    next_s, reward, done, info = env.step(config.ACTION_LEFT)
    print(
        f"    -> Out-of-bounds LEFT : State={next_s} | "
        f"Reward={reward:+5.1f} (Penalty) | Event={info['event']} | Valid={info['is_valid']}"
    )

    # C) Move to (0, 3) and try moving RIGHT into obstacle at (0, 4)
    env.step(config.ACTION_RIGHT)  # -> (0, 1)
    env.step(config.ACTION_RIGHT)  # -> (0, 2)
    env.step(config.ACTION_RIGHT)  # -> (0, 3)
    print(f"    Moved agent to {env.get_state()} (adjacent to obstacle at (0, 4))")
    next_s, reward, done, info = env.step(config.ACTION_RIGHT)
    print(
        f"    -> Obstacle Collision: State={next_s} | "
        f"Reward={reward:+5.1f} (Penalty) | Event={info['event']} | Valid={info['is_valid']}"
    )

    # 6. Goal Detection
    print("\n[6] Testing Goal Detection:")
    # Place agent right above the goal at (8, 9)
    env.agent_pos = (8, 9)
    print(f"    Positioned agent at {env.get_state()} (directly north of Goal {env.goal_pos})")
    next_s, reward, done, info = env.step(config.ACTION_DOWN)
    print(
        f"    -> Stepped DOWN to Goal: State={next_s} | "
        f"Reward={reward:+5.1f} (Success!) | Done={done} | Event={info['event']}"
    )

    print("\n[+] Phase 1 Campus Environment successfully verified and fully functional!")
    print("=" * 72)


if __name__ == "__main__":
    run_environment_demo()
