"""
Campus Navigator: An Intelligent Path-Finding Game Using Q-Learning
===================================================================
Configuration Module (config.py)

This file contains all global constants, hyperparameter definitions,
reward values, campus layout settings, and display parameters.
Keeping them centralized makes it easy to experiment with different
learning rates, reward structures, or grid layouts without modifying
the core algorithm logic.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

# ==============================================================================
# 1. DIRECTORY PATHS
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# File paths for saving learned Q-tables, training metrics, and logs
Q_TABLE_SAVE_PATH = os.path.join(MODELS_DIR, "q_table.npy")
Q_TABLE_MULTI_SAVE_PATH = os.path.join(MODELS_DIR, "q_table_multi.npy")
METRICS_CSV_PATH = os.path.join(RESULTS_DIR, "training_results.csv")
TRAINING_RESULTS_PATH = os.path.join(RESULTS_DIR, "training_results.csv")
TRAINING_RESULTS_SAVE_PATH = TRAINING_RESULTS_PATH
EVAL_RESULTS_PATH = os.path.join(RESULTS_DIR, "evaluation_summary.json")
EVAL_SUMMARY_PATH = os.path.join(RESULTS_DIR, "evaluation_summary.json")
EVAL_CSV_PATH = os.path.join(RESULTS_DIR, "evaluation_results.csv")
EXPERIMENTS_CSV_PATH = os.path.join(RESULTS_DIR, "experiments.csv")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
EXPERIMENT_COMPARISON_PLOT_PATH = os.path.join(PLOTS_DIR, "experiment_comparison.png")
REWARD_EXPERIMENTS_CSV_PATH = os.path.join(RESULTS_DIR, "reward_experiments.csv")
REWARD_COMPARISON_PLOT_PATH = os.path.join(PLOTS_DIR, "reward_profile_comparison.png")
REWARD_CURVES_PLOT_PATH = os.path.join(PLOTS_DIR, "reward_learning_curves.png")
REWARD_ROUTES_PLOT_PATH = os.path.join(PLOTS_DIR, "reward_routes_comparison.png")
FINAL_REPORT_JSON_PATH = os.path.join(RESULTS_DIR, "final_report.json")
FINAL_RESULTS_CSV_PATH = os.path.join(RESULTS_DIR, "final_results.csv")
LOG_FILE_PATH = os.path.join(LOGS_DIR, "campus_navigator.log")

# ==============================================================================
# 2. VIRTUAL CAMPUS ENVIRONMENT SETTINGS (10x10 Gridworld)
# ==============================================================================
GRID_ROWS: int = 10
GRID_COLS: int = 10

# Cell Type Identifiers
CELL_WALKABLE: int = 0        # Walkable pathway (ASCII: '.')
CELL_OBSTACLE: int = 1        # Impassable building/construction (ASCII: '#')
CELL_START: int = 2           # Starting landmark - Main Gate (ASCII: 'S')
CELL_GOAL: int = 3            # Target destination - Central Library (ASCII: 'G')
CELL_SAFE_ZONE: int = 4       # Safe student lounge/hub (ASCII: 'Z')
CELL_UNSAFE_ZONE: int = 5     # Slippery/hazard corridor (ASCII: '!')
CELL_CROWDED_ZONE: int = 6    # High-traffic congested corridor (ASCII: 'C')

# ASCII Representation Mapping
ASCII_TILES: Dict[int, str] = {
    CELL_WALKABLE: ".",
    CELL_OBSTACLE: "#",
    CELL_START: "S",
    CELL_GOAL: "G",
    CELL_SAFE_ZONE: "Z",
    CELL_UNSAFE_ZONE: "!",
    CELL_CROWDED_ZONE: "C",
}

# Start and Goal Coordinates (Default Baseline)
START_POSITION: Tuple[int, int] = (0, 0)   # Top-Left: Main Campus Gate
GOAL_POSITION: Tuple[int, int] = (9, 9)    # Bottom-Right: Central Library

# Predefined Multiple Destinations (Phase 16)
DEST_LIBRARY: str = "Library"
DEST_MAIN_BLOCK: str = "Main Block"
DEST_COMPUTER_LAB: str = "Computer Lab"
DEST_CAFETERIA: str = "Cafeteria"
DEST_SPORTS_GROUND: str = "Sports Ground"

DESTINATIONS: Dict[str, Tuple[int, int]] = {
    DEST_LIBRARY: (9, 9),
    DEST_MAIN_BLOCK: (1, 8),
    DEST_COMPUTER_LAB: (4, 3),
    DEST_CAFETERIA: (3, 2),
    DEST_SPORTS_GROUND: (9, 0),
}

# Predefined Multiple Start Locations (Phase 16)
START_MAIN_GATE: str = "Main Gate"
START_NORTH_GATE: str = "North Gate"
START_WEST_GATE: str = "West Gate"
START_HOSTEL_PLAZA: str = "Hostel Plaza"
START_CAFETERIA: str = "Cafeteria"

START_LOCATIONS: Dict[str, Tuple[int, int]] = {
    START_MAIN_GATE: (0, 0),
    START_NORTH_GATE: (0, 9),
    START_WEST_GATE: (9, 1),
    START_HOSTEL_PLAZA: (7, 4),
    START_CAFETERIA: (3, 2),
}

# Landmark Badge Labels for Procedural Grid Markers
DESTINATION_BADGES: Dict[str, str] = {
    DEST_LIBRARY: "LIB",
    DEST_MAIN_BLOCK: "MAIN",
    DEST_COMPUTER_LAB: "LAB",
    DEST_CAFETERIA: "CAFE",
    DEST_SPORTS_GROUND: "SPORT",
}

START_BADGES: Dict[str, str] = {
    START_MAIN_GATE: "GATE",
    START_NORTH_GATE: "NORTH",
    START_WEST_GATE: "WEST",
    START_HOSTEL_PLAZA: "HOSTL",
    START_CAFETERIA: "CAFE",
}

# Deterministic Campus Obstacle Layout (10x10)
# Matches the conceptual layout:
# S . . . # . . . . .
# . . # . # . . # . .
# . # # . . . . # . .
# . . . . # . . . . .
# # # . . . . # # . .
# . . . # . . . . . .
# . . . . . # . # . .
# . # . . . . . # . .
# . . . # . . . . . .
# . . . . . . # . . G
DEFAULT_OBSTACLES: List[Tuple[int, int]] = [
    (0, 4),
    (1, 2), (1, 4), (1, 7),
    (2, 1), (2, 2), (2, 7),
    (3, 4),
    (4, 0), (4, 1), (4, 6), (4, 7),
    (5, 3),
    (6, 5), (6, 7),
    (7, 1), (7, 7),
    (8, 3),
    (9, 6),
]

# Designated Safe Zones (Student Help Desks / Well-lit cafeterias - small positive reward)
DEFAULT_SAFE_ZONES: List[Tuple[int, int]] = [
    (3, 2),
    (7, 3),
]

# Designated Crowded Zones (High-density pedestrian congestion / busy corridors - small negative reward)
DEFAULT_CROWDED_ZONES: List[Tuple[int, int]] = [
    (3, 5),
    (6, 3),
    (7, 5),
]

# Designated Unsafe Zones (Rough construction fringes / waterlogged paths - larger negative reward)
DEFAULT_UNSAFE_ZONES: List[Tuple[int, int]] = [
    (2, 5),
    (5, 6),
    (8, 7),
]

# Action Space: 4 Cardinal Directions
# 0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT
ACTION_UP: int = 0
ACTION_DOWN: int = 1
ACTION_LEFT: int = 2
ACTION_RIGHT: int = 3

ACTIONS: Dict[int, str] = {
    ACTION_UP: "UP",
    ACTION_DOWN: "DOWN",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
}

ACTION_DELTA: Dict[int, Tuple[int, int]] = {
    ACTION_UP: (-1, 0),     # Move one row up
    ACTION_DOWN: (1, 0),    # Move one row down
    ACTION_LEFT: (0, -1),   # Move one column left
    ACTION_RIGHT: (0, 1),   # Move one column right
}
NUM_ACTIONS: int = len(ACTIONS)

# ==============================================================================
# 3. REWARD SYSTEM (Reward Engineering)
# ==============================================================================
REWARD_GOAL: float = 100.0                  # Destination reached successfully
REWARD_NORMAL_STEP: float = -1.0           # Baseline step cost for equal distance
REWARD_MOVE_CLOSER: float = 5.0            # Shaped reward for getting closer to destination
REWARD_MOVE_FARTHER: float = -3.0          # Shaped penalty for moving away from destination
REWARD_INVALID_MOVE: float = -10.0         # Collision with wall or impassable obstacle
REWARD_SAFE_ZONE: float = 2.0              # Small positive reward bonus for safe student hub
REWARD_CROWDED_ZONE: float = -2.0          # Small negative penalty for congested corridors
REWARD_UNSAFE_ZONE: float = -10.0          # Larger negative penalty for hazard/mud zones
REWARD_MAX_STEPS_EXCEEDED: float = -20.0   # Failure to reach goal within episode limit

# Maximum steps allowed per episode before truncation
MAX_STEPS_PER_EPISODE: int = 200

# ==============================================================================
# 4. TABULAR Q-LEARNING HYPERPARAMETERS
# ==============================================================================
LEARNING_RATE: float = 0.1          # Alpha (learning rate)
DISCOUNT_FACTOR: float = 0.9        # Gamma (discount factor: 0.9)
EPSILON_INITIAL: float = 1.0        # Initial exploration probability
EPSILON_MIN: float = 0.05           # Minimum exploration floor: 0.05
EPSILON_DECAY: float = 0.995        # Exploration decay multiplier per episode
DEFAULT_TRAINING_EPISODES: int = 1000

# ==============================================================================
# 5. PYGAME DISPLAY & RENDERING CONFIGURATION
# ==============================================================================
SCREEN_WIDTH: int = 1100
SCREEN_HEIGHT: int = 750
GRID_CELL_SIZE: int = 60            # 60px * 10 = 600px total grid width/height
FPS: int = 60
RENDER_SPEED_TRAINING: int = 120
RENDER_SPEED_PLAY: int = 6          # Autonomous step rate (steps/sec)

# Color Palette (RGB Tuples) - Clean Educational Aesthetic
COLOR_BACKGROUND: Tuple[int, int, int] = (241, 245, 249)      # Soft slate-50
COLOR_GRID_LINE: Tuple[int, int, int] = (203, 213, 225)       # Slate-300 borders
COLOR_PATHWAY: Tuple[int, int, int] = (255, 255, 255)         # Pure white walkable tiles
COLOR_OBSTACLE: Tuple[int, int, int] = (30, 41, 59)           # Dark slate-800 for buildings
COLOR_OBSTACLE_DETAIL: Tuple[int, int, int] = (71, 85, 105)   # Slate-600 windows/lines
COLOR_START: Tuple[int, int, int] = (34, 197, 94)             # Vibrant green start marker
COLOR_GOAL: Tuple[int, int, int] = (234, 179, 8)              # Rich gold target marker
COLOR_SAFE_ZONE: Tuple[int, int, int] = (99, 102, 241)        # Indigo-500 for safe zone
COLOR_CROWDED_ZONE: Tuple[int, int, int] = (245, 158, 11)     # Amber-500 for crowded zone
COLOR_UNSAFE_ZONE: Tuple[int, int, int] = (249, 115, 22)      # Amber-orange for hazard zone
COLOR_AGENT: Tuple[int, int, int] = (37, 99, 235)             # Royal blue for student agent
COLOR_AGENT_BORDER: Tuple[int, int, int] = (29, 78, 216)      # Darker blue ring
COLOR_OPTIMAL_PATH: Tuple[int, int, int] = (168, 85, 247)     # Purple path trajectory
COLOR_PANEL_BG: Tuple[int, int, int] = (15, 23, 42)           # Dark navy slate-900 HUD panel
COLOR_PANEL_CARD: Tuple[int, int, int] = (30, 41, 59)         # Slate-800 card bg
COLOR_TEXT_LIGHT: Tuple[int, int, int] = (248, 249, 250)      # Bright white text
COLOR_TEXT_MUTED: Tuple[int, int, int] = (148, 163, 184)      # Muted slate text
COLOR_ACCENT: Tuple[int, int, int] = (45, 212, 191)           # Bright teal accent
COLOR_SUCCESS: Tuple[int, int, int] = (34, 197, 94)           # Green for success
COLOR_DANGER: Tuple[int, int, int] = (239, 68, 68)            # Red for collision/penalty

# Professional UI Enhancement Accents (Phase 17)
COLOR_AXIS_RULER: Tuple[int, int, int] = (100, 116, 139)      # Slate-500 for grid axis coordinates
COLOR_AXIS_BG: Tuple[int, int, int] = (241, 245, 249)         # Soft slate background behind rulers
COLOR_STATUS_ACTIVE: Tuple[int, int, int] = (34, 197, 94)     # Emerald dot for live policy execution
COLOR_STATUS_TRAINING: Tuple[int, int, int] = (245, 158, 11)  # Amber dot for training/exploration
COLOR_STATUS_MANUAL: Tuple[int, int, int] = (56, 189, 248)    # Sky blue dot for human control
COLOR_STATUS_ALERT: Tuple[int, int, int] = (239, 68, 68)      # Crimson dot for collision/hazard
COLOR_PROGRESS_BG: Tuple[int, int, int] = (30, 41, 59)        # Dark navy progress track
COLOR_PROGRESS_FILL: Tuple[int, int, int] = (37, 99, 235)     # Vibrant blue episode progress fill
COLOR_PROGRESS_ACCENT: Tuple[int, int, int] = (96, 165, 250)  # Lighter blue progress bar highlight
COLOR_EPSILON_FILL: Tuple[int, int, int] = (147, 51, 234)     # Purple exploration decay fill
COLOR_SUCCESS_PULSE: Tuple[int, int, int] = (245, 158, 11)    # Gold destination ripple halo
COLOR_ALERT_BG: Tuple[int, int, int] = (254, 242, 242)        # Soft red failure notification banner
COLOR_ALERT_BORDER: Tuple[int, int, int] = (252, 165, 165)    # Light red border for failure banner
COLOR_ALERT_TEXT: Tuple[int, int, int] = (185, 28, 28)        # Dark crimson failure notification text

# ==============================================================================
# 6. DYNAMIC CONFIGURATION & PERSISTENCE (config.json) - PHASE 14
# ==============================================================================
DEFAULT_TRAINING_SPEED: str = "NORMAL"  # "SLOW", "NORMAL", "FAST"
CONFIG_PATH: str = os.path.join(BASE_DIR, "config.json")


def get_default_config() -> Dict:
    """Return factory default configuration settings."""
    return {
        "grid_size": 10,
        "training_episodes": 1000,
        "learning_rate": 0.1,
        "discount_factor": 0.9,
        "initial_epsilon": 1.0,
        "min_epsilon": 0.05,
        "epsilon_decay": 0.995,
        "max_steps_per_episode": 200,
        "training_speed": "NORMAL",
        "num_actions": 4,
        "reward_safe_zone": 2.0,
        "reward_crowded_zone": -2.0,
        "reward_unsafe_zone": -10.0,
    }


DEFAULT_CONFIG: Dict = get_default_config()


def validate_config(cfg: Dict) -> Tuple[bool, str]:
    """
    Validate all configuration settings against algorithmic and physical bounds.

    Explicitly rejects:
    - Negative learning rate
    - Discount factor > 1
    - Negative episodes
    - Zero actions
    - Out-of-bounds exploration, decay, and grid size

    Returns:
        (is_valid, message): (True, "OK") or (False, descriptive error message).
    """
    if not isinstance(cfg, dict):
        return False, "Configuration must be a JSON dictionary object."

    # 1. Zero Actions Check
    num_actions = cfg.get("num_actions", NUM_ACTIONS)
    if not isinstance(num_actions, int) or num_actions <= 0:
        return False, "Number of actions must be greater than zero (zero actions prohibited)."

    # 2. Grid Size Validation (Safely Supported: 6 to 16)
    grid_size = cfg.get("grid_size", GRID_ROWS)
    if not isinstance(grid_size, int):
        return False, "Grid size must be an integer."
    if grid_size < 6 or grid_size > 16:
        return False, "Grid size must be between 6 and 16 (10x10 standard campus layout)."

    # 3. Training Episodes Validation
    episodes = cfg.get("training_episodes", DEFAULT_TRAINING_EPISODES)
    if not isinstance(episodes, int):
        return False, "Training episodes must be an integer."
    if episodes <= 0:
        return False, "Training episodes cannot be negative or zero (must be > 0)."
    if episodes > 50000:
        return False, "Training episodes cannot exceed 50,000."

    # 4. Learning Rate (Alpha) Validation
    lr = cfg.get("learning_rate", LEARNING_RATE)
    if not isinstance(lr, (int, float)):
        return False, "Learning rate must be a numeric value."
    if lr < 0:
        return False, "Learning rate cannot be negative (must be between 0.001 and 1.0)."
    if lr == 0:
        return False, "Learning rate cannot be zero (agent cannot update Q-values)."
    if lr > 1.0:
        return False, "Learning rate cannot exceed 1.0 (causes Q-value divergence)."

    # 5. Discount Factor (Gamma) Validation
    df = cfg.get("discount_factor", DISCOUNT_FACTOR)
    if not isinstance(df, (int, float)):
        return False, "Discount factor must be a numeric value."
    if df < 0:
        return False, "Discount factor cannot be negative."
    if df > 1.0:
        return False, "Discount factor cannot exceed 1.0 (must be between 0.0 and 1.0)."

    # 6. Exploration Rates (Initial & Minimum Epsilon)
    eps_init = cfg.get("initial_epsilon", EPSILON_INITIAL)
    if not isinstance(eps_init, (int, float)):
        return False, "Initial epsilon must be a numeric value."
    if eps_init < 0.0 or eps_init > 1.0:
        return False, "Initial epsilon must be between 0.0 and 1.0."

    eps_min = cfg.get("min_epsilon", EPSILON_MIN)
    if not isinstance(eps_min, (int, float)):
        return False, "Minimum epsilon must be a numeric value."
    if eps_min < 0.0 or eps_min > 1.0:
        return False, "Minimum epsilon must be between 0.0 and 1.0."
    if eps_min > eps_init:
        return False, "Minimum epsilon cannot be greater than initial epsilon."

    # 7. Epsilon Decay Factor Validation
    decay = cfg.get("epsilon_decay", EPSILON_DECAY)
    if not isinstance(decay, (int, float)):
        return False, "Epsilon decay must be a numeric value."
    if decay <= 0.0 or decay > 1.0:
        return False, "Epsilon decay must be strictly between 0.0 and 1.0."
    if decay < 0.5:
        return False, "Epsilon decay is too rapid (must be >= 0.5)."

    # 8. Maximum Steps Per Episode Validation
    max_steps = cfg.get("max_steps_per_episode", MAX_STEPS_PER_EPISODE)
    if not isinstance(max_steps, int):
        return False, "Maximum steps per episode must be an integer."
    if max_steps <= 0:
        return False, "Maximum steps per episode must be a positive integer (> 0)."
    if max_steps < 10 or max_steps > 2000:
        return False, "Maximum steps per episode must be between 10 and 2000."

    # 9. Training Speed Validation
    speed = cfg.get("training_speed", DEFAULT_TRAINING_SPEED)
    if not isinstance(speed, str) or speed.upper() not in ("SLOW", "NORMAL", "FAST"):
        return False, "Training speed must be one of: 'SLOW', 'NORMAL', or 'FAST'."

    # 10. Special Zone Rewards Validation (Phase 15: Configurable Safe, Crowded, Unsafe Zones)
    if "reward_safe_zone" in cfg:
        rs = cfg["reward_safe_zone"]
        if not isinstance(rs, (int, float)):
            return False, "Safe zone reward must be a numeric value."
        if rs < 0:
            return False, "Safe zone reward must be non-negative (>= 0.0 bonus)."
        if rs > 50.0:
            return False, "Safe zone reward cannot exceed 50.0."

    if "reward_crowded_zone" in cfg:
        rc = cfg["reward_crowded_zone"]
        if not isinstance(rc, (int, float)):
            return False, "Crowded zone penalty must be a numeric value."
        if rc > 0:
            return False, "Crowded zone reward must be non-positive (<= 0.0 penalty)."
        if rc < -50.0:
            return False, "Crowded zone penalty cannot be less than -50.0."

    if "reward_unsafe_zone" in cfg:
        ru = cfg["reward_unsafe_zone"]
        if not isinstance(ru, (int, float)):
            return False, "Unsafe zone penalty must be a numeric value."
        if ru > 0:
            return False, "Unsafe zone reward must be non-positive (<= 0.0 penalty)."
        if ru < -100.0:
            return False, "Unsafe zone penalty cannot be less than -100.0."

    return True, "Configuration is valid."


def apply_config(cfg: Dict) -> None:
    """Apply validated dictionary values to active module globals."""
    global GRID_ROWS, GRID_COLS, GRID_CELL_SIZE, GOAL_POSITION
    global LEARNING_RATE, DISCOUNT_FACTOR, EPSILON_INITIAL, EPSILON_MIN, EPSILON_DECAY
    global DEFAULT_TRAINING_EPISODES, MAX_STEPS_PER_EPISODE, DEFAULT_TRAINING_SPEED
    global NUM_ACTIONS
    global REWARD_SAFE_ZONE, REWARD_CROWDED_ZONE, REWARD_UNSAFE_ZONE

    if "grid_size" in cfg:
        gs = int(cfg["grid_size"])
        GRID_ROWS = gs
        GRID_COLS = gs
        GRID_CELL_SIZE = max(30, min(80, 600 // gs))
        GOAL_POSITION = (gs - 1, gs - 1)

    if "learning_rate" in cfg:
        LEARNING_RATE = float(cfg["learning_rate"])
    if "discount_factor" in cfg:
        DISCOUNT_FACTOR = float(cfg["discount_factor"])
    if "initial_epsilon" in cfg:
        EPSILON_INITIAL = float(cfg["initial_epsilon"])
    if "min_epsilon" in cfg:
        EPSILON_MIN = float(cfg["min_epsilon"])
    if "epsilon_decay" in cfg:
        EPSILON_DECAY = float(cfg["epsilon_decay"])
    if "training_episodes" in cfg:
        DEFAULT_TRAINING_EPISODES = int(cfg["training_episodes"])
    if "max_steps_per_episode" in cfg:
        MAX_STEPS_PER_EPISODE = int(cfg["max_steps_per_episode"])
    if "training_speed" in cfg:
        DEFAULT_TRAINING_SPEED = str(cfg["training_speed"]).upper()
    if "num_actions" in cfg:
        NUM_ACTIONS = int(cfg["num_actions"])
    if "reward_safe_zone" in cfg:
        REWARD_SAFE_ZONE = float(cfg["reward_safe_zone"])
    if "reward_crowded_zone" in cfg:
        REWARD_CROWDED_ZONE = float(cfg["reward_crowded_zone"])
    if "reward_unsafe_zone" in cfg:
        REWARD_UNSAFE_ZONE = float(cfg["reward_unsafe_zone"])


def load_config(file_path: Optional[str] = None) -> Dict:
    """
    Load configuration from config.json, validate, and apply to globals.
    Falls back to factory defaults if file is missing or contains invalid values.
    """
    path = file_path or CONFIG_PATH
    if not os.path.exists(path):
        cfg = get_default_config()
        save_config(cfg, path)
        return cfg

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        valid, msg = validate_config(cfg)
        if valid:
            apply_config(cfg)
            return cfg
        else:
            try:
                from utils.logger import log_warning
                log_warning(f"Invalid config.json detected: {msg}. Falling back to default configuration.")
            except ImportError:
                pass
            print(f"[Warning] Invalid config.json detected: {msg}. Falling back to defaults.")
            defaults = get_default_config()
            apply_config(defaults)
            return defaults
    except Exception as e:
        try:
            from utils.logger import log_exception
            log_exception(e, f"load_config({path})")
        except ImportError:
            pass
        print(f"[Warning] Failed to load config from {path}: {e}. Falling back to defaults.")
        defaults = get_default_config()
        apply_config(defaults)
        return defaults


def save_config(cfg: Dict, file_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validate and save configuration dictionary to config.json.
    Updates active runtime configuration globals upon success.
    """
    valid, msg = validate_config(cfg)
    if not valid:
        try:
            from utils.logger import log_warning
            log_warning(f"Configuration validation rejected settings: {msg}")
        except ImportError:
            pass
        return False, msg

    path = file_path or CONFIG_PATH
    try:
        apply_config(cfg)
        clean_cfg = {
            "grid_size": int(cfg.get("grid_size", GRID_ROWS)),
            "training_episodes": int(cfg.get("training_episodes", DEFAULT_TRAINING_EPISODES)),
            "learning_rate": round(float(cfg.get("learning_rate", LEARNING_RATE)), 4),
            "discount_factor": round(float(cfg.get("discount_factor", DISCOUNT_FACTOR)), 4),
            "initial_epsilon": round(float(cfg.get("initial_epsilon", EPSILON_INITIAL)), 4),
            "min_epsilon": round(float(cfg.get("min_epsilon", EPSILON_MIN)), 4),
            "epsilon_decay": round(float(cfg.get("epsilon_decay", EPSILON_DECAY)), 5),
            "max_steps_per_episode": int(cfg.get("max_steps_per_episode", MAX_STEPS_PER_EPISODE)),
            "training_speed": str(cfg.get("training_speed", DEFAULT_TRAINING_SPEED)).upper(),
            "num_actions": int(cfg.get("num_actions", NUM_ACTIONS)),
            "reward_safe_zone": round(float(cfg.get("reward_safe_zone", REWARD_SAFE_ZONE)), 2),
            "reward_crowded_zone": round(float(cfg.get("reward_crowded_zone", REWARD_CROWDED_ZONE)), 2),
            "reward_unsafe_zone": round(float(cfg.get("reward_unsafe_zone", REWARD_UNSAFE_ZONE)), 2),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean_cfg, f, indent=2)
        return True, "Settings saved successfully to config.json."
    except Exception as e:
        try:
            from utils.logger import log_exception
            log_exception(e, f"save_config({path})")
        except ImportError:
            pass
        return False, f"Failed to write config.json: {str(e)}"


def reset_to_defaults(file_path: Optional[str] = None) -> Dict:
    """Reset configuration to factory defaults, save to config.json, and apply."""
    defaults = get_default_config()
    save_config(defaults, file_path)
    return defaults


# Automatically load saved configuration from config.json if it exists
if os.path.exists(CONFIG_PATH):
    try:
        load_config()
    except Exception:
        pass
