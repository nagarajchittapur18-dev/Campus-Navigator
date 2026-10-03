"""
Campus Navigator: Tabular Q-Learning Agent (Phase 2)
===================================================
Module: agent/q_learning.py

RL Concept - TABULAR Q-LEARNING AGENT:
The Agent is the decision-making entity that learns an optimal policy through
trial-and-error interaction with the campus environment.

Key Components & Theory:
------------------------
1. State-to-Index Mapping:
   Converts a 2D grid position (row, column) into a discrete 1D scalar index:
       state_index = row * total_columns + column
   Example: On a 10x10 grid, (0, 0) -> 0, (0, 5) -> 5, (1, 0) -> 10, (9, 9) -> 99.

2. Q-Table Q(s, a):
   A 2D NumPy lookup table of shape [num_states, num_actions] (100 states x 4 actions).
   Each entry Q(s, a) represents the "quality" or expected cumulative discounted
   future reward of taking action 'a' in state 's'.

3. Exploration vs. Exploitation (Epsilon-Greedy Strategy):
   - Exploration (probability epsilon): Choose a random action to discover
     unvisited pathways across campus and avoid getting trapped in local optima.
   - Exploitation (probability 1 - epsilon): Choose the greedy action with the
     highest Q-value in the current state to maximize reward.
   - Valid Actions Support: Optionally restricts choices to legal, non-colliding moves.

4. Bellman Optimality Equation (TD Update):
   Q(s, a) <- Q(s, a) + alpha * [ TD Target - Q(s, a) ]

   Where:
     - TD Target (Non-terminal): Reward + gamma * max_a' Q(s', a')
     - TD Target (Terminal state): Reward   (No future discounted rewards!)
     - TD Error: TD Target - Q(s, a)
     - alpha: Learning Rate (how quickly new experience overrides old estimates)
     - gamma: Discount Factor (how much future rewards are valued vs. immediate)

5. Epsilon Decay:
   Gradually decreases epsilon after each episode:
       epsilon = max(epsilon_min, epsilon * epsilon_decay)
   This transitions the agent from a curious explorer early in training
   into a disciplined navigator exploiting learned shortest paths.
"""

import os
import sys
from typing import List, Optional, Tuple, Union
import numpy as np

# Ensure project root is accessible when running directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config


class QLearningAgent:
    """
    Tabular Q-Learning Agent for 2D Gridworld Campus Navigation.
    """

    def __init__(
        self,
        rows: int = config.GRID_ROWS,
        cols: int = config.GRID_COLS,
        num_actions: int = config.NUM_ACTIONS,
        learning_rate: float = config.LEARNING_RATE,
        discount_factor: float = config.DISCOUNT_FACTOR,
        epsilon: float = config.EPSILON_INITIAL,
        epsilon_min: float = config.EPSILON_MIN,
        epsilon_decay: float = config.EPSILON_DECAY,
    ) -> None:
        """
        Initialize the Q-Learning Agent with tabular memory and hyperparameters.

        Args:
            rows: Total campus grid rows (default: 10).
            cols: Total campus grid columns (default: 10).
            num_actions: Total discrete actions (default: 4 -> UP, DOWN, LEFT, RIGHT).
            learning_rate (alpha): Step size for updating Q-values (default: 0.1).
            discount_factor (gamma): Future reward discount multiplier (default: 0.9).
            epsilon: Initial exploration probability (default: 1.0).
            epsilon_min: Minimum exploration floor (default: 0.05).
            epsilon_decay: Multiplicative decay rate per episode (default: 0.995).
        """
        self.rows: int = rows
        self.cols: int = cols
        self.num_states: int = rows * cols
        self.num_actions: int = num_actions

        # Hyperparameters
        self.alpha: float = learning_rate
        self.gamma: float = discount_factor
        self.epsilon: float = epsilon
        self.epsilon_initial: float = epsilon
        self.epsilon_min: float = epsilon_min
        self.epsilon_decay: float = epsilon_decay

        # Phase 16: Goal-Conditioned Multi-Destination Memory
        self.destinations: Dict[str, Tuple[int, int]] = dict(config.DESTINATIONS)
        self.dest_names: List[str] = list(self.destinations.keys())
        self.dest_coords: List[Tuple[int, int]] = list(self.destinations.values())
        self.num_destinations: int = len(self.dest_names)
        self.current_dest_id: int = 0
        self.current_dest_name: str = self.dest_names[0]

        # Multi-Destination Q-Table: Shape (num_destinations, num_states, num_actions) -> (5, 100, 4)
        self.multi_q_table: np.ndarray = np.zeros(
            (self.num_destinations, self.num_states, self.num_actions), dtype=np.float32
        )
        # self.q_table is a direct 2D slice view into multi_q_table for the active destination
        # Updating self.q_table automatically mutates multi_q_table[current_dest_id] in-place!
        self.q_table: np.ndarray = self.multi_q_table[self.current_dest_id]

        # Phase 18: Model persistence status & error diagnostics
        self.last_load_status: str = "UNLOADED"
        self.last_load_error_msg: str = ""

    def set_destination(self, dest: Union[str, int, Tuple[int, int]]) -> int:
        """
        Switch active destination for destination-aware Q-learning (Phase 16).

        Updates self.current_dest_id and re-points self.q_table to the corresponding
        2D slice of self.multi_q_table in O(1) time without data copying.

        Args:
            dest: Landmark name string ('Sports Ground'), integer ID (0..4), or coordinate (r, c).

        Returns:
            dest_id: Integer index of selected destination.
        """
        if isinstance(dest, int):
            if 0 <= dest < self.num_destinations:
                self.current_dest_id = dest
            else:
                raise ValueError(f"Destination ID {dest} out of range [0, {self.num_destinations - 1}].")
        elif isinstance(dest, str):
            if dest in self.dest_names:
                self.current_dest_id = self.dest_names.index(dest)
            else:
                raise ValueError(f"Unknown destination landmark '{dest}'. Valid: {self.dest_names}")
        elif isinstance(dest, (tuple, list)):
            coord = (int(dest[0]), int(dest[1]))
            if coord in self.dest_coords:
                self.current_dest_id = self.dest_coords.index(coord)
            else:
                self.current_dest_id = 0
        else:
            raise TypeError(f"Invalid destination format {type(dest)}.")

        self.current_dest_name = self.dest_names[self.current_dest_id]
        self.q_table = self.multi_q_table[self.current_dest_id]
        return self.current_dest_id

    def set_destination_id(self, dest_id: int) -> int:
        """Alias for set_destination with integer index."""
        return self.set_destination(dest_id)

    # --------------------------------------------------------------------------
    # 1. State-to-Index Conversion
    # --------------------------------------------------------------------------

    def state_to_index(self, state: Union[Tuple[int, int], int]) -> int:
        """
        Convert a 2D (row, column) coordinate into a unique 1D discrete state index.

        Formula:
            state_index = row * total_columns + column

        Examples on 10x10 Campus Grid:
            (0, 0) -> 0 * 10 + 0 = 0   (Start: Main Gate)
            (0, 9) -> 0 * 10 + 9 = 9
            (1, 0) -> 1 * 10 + 0 = 10
            (9, 9) -> 9 * 10 + 9 = 99  (Goal: Central Library)

        Args:
            state: Either a Tuple (row, col) or already an integer index.

        Returns:
            state_index: Integer in range [0, num_states - 1].
        """
        if isinstance(state, (tuple, list)):
            if len(state) < 2:
                err_msg = f"State tuple {state} must have at least 2 coordinate elements (row, col)."
                try:
                    from utils.logger import log_error
                    log_error(err_msg)
                except ImportError:
                    pass
                raise ValueError(err_msg)
            row, col = int(state[0]), int(state[1])
            if not (0 <= row < self.rows and 0 <= col < self.cols):
                err_msg = f"State coordinate ({row}, {col}) is out of campus bounds ({self.rows}x{self.cols})."
                try:
                    from utils.logger import log_error
                    log_error(err_msg)
                except ImportError:
                    pass
                raise ValueError(err_msg)
            return int(row * self.cols + col)
        try:
            s_int = int(state)
            if not (0 <= s_int < self.num_states):
                err_msg = f"State index {s_int} out of bounds [0, {self.num_states - 1}]."
                try:
                    from utils.logger import log_error
                    log_error(err_msg)
                except ImportError:
                    pass
                raise ValueError(err_msg)
            return s_int
        except (TypeError, ValueError) as e:
            err_msg = f"Invalid state representation: {state} (type: {type(state).__name__})."
            try:
                from utils.logger import log_error
                log_error(err_msg)
            except ImportError:
                pass
            raise ValueError(err_msg) from e

    def index_to_state(self, index: int) -> Tuple[int, int]:
        """
        Convert a 1D discrete state index back into a 2D (row, column) coordinate.

        Formula:
            row = index // total_columns
            col = index % total_columns

        Args:
            index: Integer state index in range [0, num_states - 1].

        Returns:
            state: Tuple (row, col).
        """
        if not isinstance(index, (int, np.integer)) or not (0 <= index < self.num_states):
            err_msg = f"State index {index} out of bounds [0, {self.num_states - 1}]."
            try:
                from utils.logger import log_error
                log_error(err_msg)
            except ImportError:
                pass
            raise ValueError(err_msg)
        return (int(index // self.cols), int(index % self.cols))

    # --------------------------------------------------------------------------
    # 2. Q-Value Inspection & Best Action Selection
    # --------------------------------------------------------------------------

    def get_q_values(self, state: Union[Tuple[int, int], int]) -> np.ndarray:
        """
        Return the 1D array of Q-values for all actions at the given state.

        Args:
            state: State coordinate (row, col) or integer state index.

        Returns:
            q_values: NumPy float array of shape (num_actions,) containing Q(s, a).
        """
        s_idx = self.state_to_index(state)
        return self.q_table[s_idx].copy()

    def get_best_action(
        self,
        state: Union[Tuple[int, int], int],
        valid_actions: Optional[List[int]] = None,
    ) -> int:
        """
        Select the greedy action with the highest Q-value in the given state.

        RL Concept - EXPLOITATION:
        Selects a* = argmax_a Q(s, a).
        If multiple actions tie for the maximum Q-value, ties are broken
        randomly to prevent directional bias.
        If valid_actions is specified, selection is restricted to legal moves.

        Args:
            state: State coordinate (row, col) or integer state index.
            valid_actions: Optional list of legal action indices.

        Returns:
            action: Integer action index.
        """
        s_idx = self.state_to_index(state)
        q_vals = self.q_table[s_idx]

        if valid_actions is not None:
            clean_valid = [int(a) for a in valid_actions if isinstance(a, (int, np.integer)) and 0 <= a < self.num_actions]
            if len(clean_valid) > 0:
                # Filter Q-values only for valid/passable actions
                valid_q = [q_vals[a] for a in clean_valid]
                max_q = max(valid_q)
                # Find all valid actions achieving this maximum (tie-breaking)
                best_candidates = [a for a, q in zip(clean_valid, valid_q) if q == max_q]
                return int(np.random.choice(best_candidates))
            else:
                try:
                    from utils.logger import log_warning
                    log_warning(f"Empty or invalid valid_actions passed to get_best_action ({valid_actions}). Falling back to unrestricted argmax.")
                except ImportError:
                    pass

        # Unrestricted greedy selection over all actions
        max_q = np.max(q_vals)
        best_candidates = np.where(q_vals == max_q)[0]
        return int(np.random.choice(best_candidates))

    # --------------------------------------------------------------------------
    # 3. Action Selection (Epsilon-Greedy Strategy)
    # --------------------------------------------------------------------------

    def choose_action(
        self,
        state: Union[Tuple[int, int], int],
        valid_actions: Optional[List[int]] = None,
        training: bool = True,
    ) -> int:
        """
        Select an action using the Epsilon-Greedy Exploration/Exploitation strategy.

        RL Concept - EXPLORATION VS. EXPLOITATION:
        - When training=True and with probability epsilon:
          EXPLORATION -> Pick a random action (discovers new campus shortcuts).
        - When training=False or with probability (1 - epsilon):
          EXPLOITATION -> Pick the best action with the highest learned Q-value.

        Args:
            state: Current agent coordinate (row, col) or state index.
            valid_actions: Optional list of passable action indices.
            training: If False, forces pure greedy exploitation (epsilon ignored).

        Returns:
            action: Selected action integer (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT).
        """
        # Exploration: Random exploratory choice (only when training=True)
        if training and np.random.rand() < self.epsilon:
            if valid_actions is not None:
                clean_valid = [int(a) for a in valid_actions if isinstance(a, (int, np.integer)) and 0 <= a < self.num_actions]
                if len(clean_valid) > 0:
                    return int(np.random.choice(clean_valid))
            return int(np.random.randint(0, self.num_actions))

        # Exploitation: Greedy choice using Q-table
        return self.get_best_action(state, valid_actions)

    # --------------------------------------------------------------------------
    # 4. Q-Value Update (Bellman Equation)
    # --------------------------------------------------------------------------

    def update(
        self,
        state: Union[Tuple[int, int], int],
        action: int,
        reward: float,
        next_state: Union[Tuple[int, int], int],
        done: bool,
    ) -> float:
        """
        Update Q(s, a) using the Bellman Optimality Equation.

        RL Concept - TEMPORAL DIFFERENCE (TD) UPDATE RULE:
        Q(s, a) <- Q(s, a) + alpha * [ TD Target - Q(s, a) ]

        Where:
        - TD Target (Non-terminal): reward + gamma * max_a' Q(s', a')
        - TD Target (Terminal state): reward
          (Important: For terminal states like reaching the goal, no future
           Q-value is added because the episode ends!)
        - TD Error: TD Target - Q(s, a)

        Args:
            state: Current state (row, col) or state index.
            action: Action taken from current state (0..3).
            reward: Immediate scalar reward received from the environment.
            next_state: Transitioned state (row, col) or state index.
            done: Boolean flag indicating if next_state is terminal.

        Returns:
            td_error: The calculated Temporal Difference error.
        """
        if not isinstance(action, (int, np.integer)) or not (0 <= int(action) < self.num_actions):
            err_msg = f"Invalid action {action} (type: {type(action).__name__}) passed to agent.update(). Valid actions: [0..{self.num_actions - 1}]."
            try:
                from utils.logger import log_error
                log_error(err_msg)
            except ImportError:
                pass
            return 0.0

        act_idx = int(action)
        s_idx = self.state_to_index(state)
        next_s_idx = self.state_to_index(next_state)

        current_q = float(self.q_table[s_idx, act_idx])

        # Terminal state handling: no future discounted reward
        if done:
            target = float(reward)
        else:
            best_future_q = float(np.max(self.q_table[next_s_idx]))
            target = float(reward) + self.gamma * best_future_q

        # Temporal Difference (TD) error: difference between estimate and target
        td_error = target - current_q

        # Update Q-value
        self.q_table[s_idx, act_idx] = current_q + self.alpha * td_error

        return float(td_error)

    # --------------------------------------------------------------------------
    # 5. Epsilon Decay
    # --------------------------------------------------------------------------

    def decay_epsilon(self) -> float:
        """
        Decay the exploration rate epsilon after an episode.

        Formula:
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        Returns:
            current_epsilon: Updated exploration rate.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    # --------------------------------------------------------------------------
    # 6. Persistence & Reset
    # --------------------------------------------------------------------------

    def save_model(self, path: Optional[str] = None, save_multi: bool = True) -> str:
        """
        Save the Q-table to NumPy .npy binary files.
        Maintains 100% backward compatibility by always writing single-destination table
        (Library) to config.Q_TABLE_SAVE_PATH and multi-destination table to config.Q_TABLE_MULTI_SAVE_PATH.

        Args:
            path: Target file path. Defaults to config.Q_TABLE_SAVE_PATH.
            save_multi: If True, also saves the complete (5, 100, 4) multi-destination table.

        Returns:
            saved_path: Absolute path to the saved primary model file.
        """
        target_path = path or config.Q_TABLE_SAVE_PATH
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        # Always save Destination 0 (Library) as the standard (100, 4) table
        np.save(target_path, self.multi_q_table[0])

        if save_multi:
            if path is not None:
                multi_path = os.path.join(os.path.dirname(target_path), "q_table_multi.npy")
            else:
                multi_path = getattr(config, "Q_TABLE_MULTI_SAVE_PATH", os.path.join(os.path.dirname(target_path), "q_table_multi.npy"))
            if os.path.exists(multi_path) and os.path.getsize(multi_path) > 0:
                try:
                    existing_multi = np.load(multi_path)
                    if existing_multi.shape == self.multi_q_table.shape:
                        if np.count_nonzero(self.multi_q_table[1:]) == 0 and np.count_nonzero(existing_multi[1:]) > 0:
                            existing_multi[self.current_dest_id] = self.multi_q_table[self.current_dest_id]
                            np.save(multi_path, existing_multi)
                            return target_path
                except Exception:
                    pass
            np.save(multi_path, self.multi_q_table)

        return target_path

    def load_model(self, path: Optional[str] = None) -> bool:
        """
        Load trained Q-table(s) from NumPy .npy binary file(s).
        Seamlessly handles both (100, 4) single-destination files and
        (num_destinations, 100, 4) multi-destination files.

        Args:
            path: Source file path. Defaults to config.Q_TABLE_SAVE_PATH.

        Returns:
            success: True if file exists and loaded successfully, False otherwise.
        """
        # Helper validator for loaded NumPy arrays
        def validate_q_array(arr: np.ndarray, file_label: str) -> Tuple[bool, str]:
            if not isinstance(arr, np.ndarray):
                return False, f"File {file_label} does not contain a valid NumPy ndarray (got {type(arr)})."
            if np.isnan(arr).any():
                return False, f"File {file_label} contains NaN (Not-a-Number) values (corrupted weights)."
            if np.isinf(arr).any():
                return False, f"File {file_label} contains infinite (Inf) values (diverged weights)."
            return True, "OK"

        self.last_load_status = "UNLOADED"
        self.last_load_error_msg = ""

        # Check if caller wants default models
        is_default_request = (
            path is None or
            os.path.abspath(path) == os.path.abspath(config.Q_TABLE_SAVE_PATH) or
            os.path.abspath(path) == os.path.abspath(getattr(config, "Q_TABLE_MULTI_SAVE_PATH", os.path.join(config.MODELS_DIR, "q_table_multi.npy")))
        )

        # Case A: Explicit custom path provided by caller
        if not is_default_request:
            source_path = path
            if not os.path.exists(source_path):
                self.last_load_status = "MISSING"
                self.last_load_error_msg = f"No trained model file found at '{source_path}'."
                try:
                    from utils.logger import log_warning
                    log_warning(self.last_load_error_msg)
                except ImportError:
                    pass
                return False

            if os.path.getsize(source_path) == 0:
                self.last_load_status = "EMPTY_FILE"
                self.last_load_error_msg = f"Model file '{source_path}' is completely empty (0 bytes)."
                try:
                    from utils.logger import log_error
                    log_error(self.last_load_error_msg)
                except ImportError:
                    pass
                return False

            try:
                loaded_arr = np.load(source_path)
            except Exception as e:
                self.last_load_status = "CORRUPTED"
                self.last_load_error_msg = f"Failed to load Q-table '{source_path}': {e}"
                try:
                    from utils.logger import log_exception
                    log_exception(e, f"load_model({source_path})")
                except ImportError:
                    pass
                return False

            valid_arr, msg = validate_q_array(loaded_arr, source_path)
            if not valid_arr:
                self.last_load_status = "CORRUPTED_VALUES"
                self.last_load_error_msg = msg
                try:
                    from utils.logger import log_error
                    log_error(msg)
                except ImportError:
                    pass
                return False

            if loaded_arr.shape == (self.num_states, self.num_actions):
                self.multi_q_table[self.current_dest_id] = loaded_arr.astype(np.float32)
                self.q_table = self.multi_q_table[self.current_dest_id]
                self.last_load_status = "OK"
                self.last_load_error_msg = ""
                try:
                    from utils.logger import log_info
                    log_info(f"Q-Table successfully loaded from '{source_path}'.")
                except ImportError:
                    pass
                return True
            elif loaded_arr.shape == self.multi_q_table.shape:
                self.multi_q_table = loaded_arr.astype(np.float32)
                self.q_table = self.multi_q_table[self.current_dest_id]
                self.last_load_status = "OK"
                self.last_load_error_msg = ""
                try:
                    from utils.logger import log_info
                    log_info(f"Multi-destination Q-Table successfully loaded from '{source_path}'.")
                except ImportError:
                    pass
                return True
            else:
                self.last_load_status = "INVALID_SHAPE"
                self.last_load_error_msg = (
                    f"Q-table shape mismatch: got {loaded_arr.shape}, expected {(self.num_states, self.num_actions)} or {self.multi_q_table.shape}."
                )
                try:
                    from utils.logger import log_error
                    log_error(self.last_load_error_msg)
                except ImportError:
                    pass
                return False

        # Case B: Default paths from config (path is None)
        multi_path = getattr(config, "Q_TABLE_MULTI_SAVE_PATH", os.path.join(config.MODELS_DIR, "q_table_multi.npy"))
        single_path = config.Q_TABLE_SAVE_PATH

        if not os.path.exists(multi_path) and not os.path.exists(single_path):
            self.last_load_status = "MISSING"
            self.last_load_error_msg = f"No trained model file found at '{single_path}' or '{multi_path}'."
            try:
                from utils.logger import log_warning
                log_warning(self.last_load_error_msg)
            except ImportError:
                pass
            return False

        if os.path.exists(multi_path) and os.path.getsize(multi_path) > 0:
            try:
                loaded_multi = np.load(multi_path)
                valid_arr, msg = validate_q_array(loaded_multi, multi_path)
                if valid_arr and loaded_multi.shape == self.multi_q_table.shape:
                    self.multi_q_table = loaded_multi.astype(np.float32)
                    self.q_table = self.multi_q_table[self.current_dest_id]
                    self.last_load_status = "OK"
                    self.last_load_error_msg = ""
                    try:
                        from utils.logger import log_info
                        log_info(f"Multi-destination Q-Table loaded from default '{multi_path}'.")
                    except ImportError:
                        pass
                    return True
            except Exception:
                pass

        if os.path.exists(single_path) and os.path.getsize(single_path) > 0:
            try:
                loaded_single = np.load(single_path)
                valid_arr, msg = validate_q_array(loaded_single, single_path)
                if valid_arr and loaded_single.shape == (self.num_states, self.num_actions):
                    self.multi_q_table[self.current_dest_id] = loaded_single.astype(np.float32)
                    self.q_table = self.multi_q_table[self.current_dest_id]
                    self.last_load_status = "OK"
                    self.last_load_error_msg = ""
                    try:
                        from utils.logger import log_info
                        log_info(f"Single-destination Q-Table loaded from default '{single_path}'.")
                    except ImportError:
                        pass
                    return True
            except Exception:
                pass

        self.last_load_status = "CORRUPTED"
        self.last_load_error_msg = f"Could not load valid Q-table from defaults '{single_path}' or '{multi_path}'."
        return False

    def reset(self) -> None:
        """
        Reset the agent's knowledge and hyperparameters back to initial state.
        Zeros out the multi-destination Q-table and restores initial epsilon.
        """
        self.multi_q_table.fill(0.0)
        self.q_table = self.multi_q_table[self.current_dest_id]
        self.epsilon = self.epsilon_initial

    # Backwards-compatible aliases
    save_q_table = save_model
    load_q_table = load_model


# ==============================================================================
# Standalone Demonstration & Test Suite
# ==============================================================================

def run_agent_demo() -> None:
    """
    Demonstrate all Phase 2 Agent capabilities:
    1. State-to-index conversion
    2. Random exploration (epsilon = 1.0)
    3. Greedy exploitation (epsilon = 0.0)
    4. Q-value update via Bellman equation
    5. Epsilon decay
    6. Model save and load
    """
    print("=" * 72)
    print(" CAMPUS NAVIGATOR: PHASE 2 Q-LEARNING AGENT DEMO & VERIFICATION")
    print("=" * 72)

    # 1. Initialization
    agent = QLearningAgent(
        rows=10,
        cols=10,
        num_actions=4,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    )
    print(f"\n[1] Agent Initialized:")
    print(f"    - Grid Size        : {agent.rows}x{agent.cols} -> {agent.num_states} states")
    print(f"    - Actions          : {agent.num_actions} (0:UP, 1:DOWN, 2:LEFT, 3:RIGHT)")
    print(f"    - Learning Rate (alpha): {agent.alpha}")
    print(f"    - Discount Factor(gamma): {agent.gamma}")
    print(f"    - Epsilon (epsilon): {agent.epsilon} (Min: {agent.epsilon_min}, Decay: {agent.epsilon_decay})")
    print(f"    - Q-Table Shape    : {agent.q_table.shape} initialized with zeros.")

    # 2. State-to-Index Conversion
    print(f"\n[2] State-to-Index & Index-to-State Mapping:")
    test_coords = [(0, 0), (0, 5), (1, 0), (4, 7), (9, 9)]
    for coord in test_coords:
        idx = agent.state_to_index(coord)
        recovered = agent.index_to_state(idx)
        print(f"    Coordinate {coord} -> State Index {idx:2d} -> Recovered Coordinate {recovered}")

    # 3. Random Exploration (Epsilon = 1.0)
    print(f"\n[3] Testing Exploration (epsilon = 1.0):")
    agent.epsilon = 1.0
    sample_state = (0, 0)
    chosen_actions = [agent.choose_action(sample_state) for _ in range(10)]
    action_names = [config.ACTIONS[a] for a in chosen_actions]
    print(f"    10 exploratory actions from {sample_state}:")
    print(f"    {action_names}")
    print(f"    -> Demonstrates random sampling across all available moves.")

    # 4. Q-Value Update Demonstration (Bellman Equation)
    print(f"\n[4] Testing Q-Value Update via Bellman Optimality Equation:")
    state = (0, 0)
    action = config.ACTION_DOWN  # Action 1 (Move DOWN)
    reward = +5.0               # Moved closer to goal
    next_state = (1, 0)
    done = False

    print(f"    Transition : {state} taking action {config.ACTIONS[action]} -> {next_state}")
    print(f"    Reward     : {reward:+5.1f} | Done: {done}")
    print(f"    Q(s, a) Before Update: {agent.get_q_values(state)[action]:.4f}")

    # Step 1 update
    td_error = agent.update(state, action, reward, next_state, done)
    print(f"    TD Error   : {td_error:+.4f}")
    print(f"    Q(s, a) After Update : {agent.get_q_values(state)[action]:.4f}")
    print(f"    Formula Calculation: 0.0 + 0.1 * (5.0 + 0.9 * 0.0 - 0.0) = 0.5")

    # Step 2: Simulate next_state having a learned high value
    next_s_idx = agent.state_to_index(next_state)
    agent.q_table[next_s_idx, config.ACTION_RIGHT] = 20.0
    print(f"\n    Simulating next state {next_state} having max Q = 20.0:")
    td_error_2 = agent.update(state, action, reward, next_state, done)
    print(f"    Updated Q(s, a) with future reward propagation: {agent.get_q_values(state)[action]:.4f}")
    print(f"    Formula Calculation: 0.5 + 0.1 * [5.0 + 0.9 * (20.0) - 0.5] = 0.5 + 0.1 * (22.5) = 2.75")

    # 5. Greedy Exploitation (Epsilon = 0.0)
    print(f"\n[5] Testing Greedy Exploitation (epsilon = 0.0):")
    agent.epsilon = 0.0
    best_act = agent.choose_action(state)
    print(f"    Q-values at {state}: {agent.get_q_values(state)}")
    print(f"    Greedy Action Selected: {best_act} ({config.ACTIONS[best_act]})")
    assert best_act == config.ACTION_DOWN, "Greedy action should choose the highest Q-value action!"
    print(f"    -> Successfully exploits maximum learned Q-value without randomness!")

    # 6. Terminal State Handling (No Future Q Added)
    print(f"\n[6] Testing Terminal State Update (Goal Reached):")
    goal_state = (8, 9)
    action_to_goal = config.ACTION_DOWN
    goal_next_state = (9, 9)
    goal_reward = +100.0
    is_done = True

    td_error_term = agent.update(goal_state, action_to_goal, goal_reward, goal_next_state, is_done)
    print(f"    Terminal Update from {goal_state} to {goal_next_state}:")
    print(f"    TD Error: {td_error_term:+.4f} | New Q(s, a): {agent.get_q_values(goal_state)[action_to_goal]:.4f}")
    print(f"    -> Target is strictly immediate reward ({goal_reward}); no future Q is added.")

    # 7. Epsilon Decay Demonstration
    print(f"\n[7] Testing Epsilon Decay Schedule:")
    agent.epsilon = 1.0
    print(f"    Starting Epsilon : {agent.epsilon:.4f}")
    for _ in range(50):
        agent.decay_epsilon()
    print(f"    After 50 episodes: {agent.epsilon:.4f}")
    for _ in range(550):
        agent.decay_epsilon()
    print(f"    After 600 episodes: {agent.epsilon:.4f} (Floored at epsilon_min: {agent.epsilon_min})")

    # 8. Model Save and Load
    print(f"\n[8] Testing Model Save & Load:")
    save_path = os.path.join(config.MODELS_DIR, "demo_test_q_table.npy")
    saved_file = agent.save_model(save_path)
    print(f"    Saved Q-table to: {saved_file}")

    new_agent = QLearningAgent(rows=10, cols=10)
    print(f"    New Agent Q-value before loading: {new_agent.get_q_values(state)[action]:.4f}")
    success = new_agent.load_model(save_path)
    print(f"    Loaded from file (Success: {success})")
    print(f"    New Agent Q-value after loading : {new_agent.get_q_values(state)[action]:.4f}")
    assert new_agent.get_q_values(state)[action] == agent.get_q_values(state)[action]
    print(f"    -> Model save and restore verified perfectly!")

    # Clean up trial demo file
    if os.path.exists(save_path):
        os.remove(save_path)

    print("\n[+] Phase 2 Q-Learning Agent successfully verified and fully functional!")
    print("=" * 72)


if __name__ == "__main__":
    run_agent_demo()
