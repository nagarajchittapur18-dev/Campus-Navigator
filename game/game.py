"""
Campus Navigator: Main Game Controller (Phase 8 - AI Training Mode)
===================================================================
Module: game/game.py

Orchestrates the entire educational Pygame application with:
- Menu Mode (Phase 7): 6 interactive options with dual mouse/keyboard controls.
- AI Training Mode (Phase 8):
    1. Initializes a fresh Q-table (clears prior knowledge).
    2. Runs training episodes with live animated agent movement.
    3. Live telemetry: Episode, Reward, Steps, Epsilon, Success/Failure, Collisions.
    4. Interactive training speed controls: Slow, Normal, Fast.
    5. Non-blocking UI with safe frame-driven stepping (no race conditions/crashes).
    6. "Stop Training" button cleanly finalizes learning early if desired.
    7. Automatically saves Q-table, logs CSV metrics, and generates plots.
    8. Shows an attractive Training Completion Screen.
- Watch Trained AI Mode: Autonomous greedy navigation following the learned policy.
- Play Game Mode: Manual human navigation to experience step costs and obstacles.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pygame

# Ensure project root is accessible when running standalone
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config
from environment.campus_env import CampusEnvironment
from agent.q_learning import QLearningAgent
from game.renderer import CampusRenderer
from game.menu import GameMenu, Button
from training.evaluate import evaluate_agent
from visualization.plots import generate_all_plots


class CampusNavigatorGame:
    """Core 2D Pygame Controller for Campus Navigator."""

    def __init__(self) -> None:
        """Initialize Pygame display, environment, agent, and renderer."""
        try:
            pygame.init()
            pygame.display.set_caption("Campus Navigator: Q-Learning Path Finding")
            self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
        except (pygame.error, Exception) as e:
            try:
                from utils.logger import log_exception
                log_exception(e, "CampusNavigatorGame.__init__ (Pygame display initialization)")
            except ImportError:
                pass
            print("\n" + "=" * 76)
            print(" [ERROR] PYGAME DISPLAY INITIALIZATION FAILED")
            print("=" * 76)
            print(f" Reason: {e}")
            print("\n This usually occurs when running in a headless or remote server environment")
            print(" without an active video display device or monitor.")
            print("\n Friendly Alternatives:")
            print(" 1. Run headless Q-Learning training in the terminal:")
            print("      python main.py --train --episodes 600")
            print(" 2. Run policy evaluation in the terminal:")
            print("      python main.py --eval")
            print(" 3. If running headless tests, configure a dummy display driver:")
            print("      Windows CMD:         set SDL_VIDEODRIVER=dummy")
            print("      Windows PowerShell:  $env:SDL_VIDEODRIVER='dummy'")
            print("      Linux / macOS:       export SDL_VIDEODRIVER=dummy")
            print("=" * 76 + "\n")
            raise RuntimeError(f"Pygame graphical initialization failed: {e}") from e

        # Core decoupled modules
        self.env = CampusEnvironment(
            rows=config.GRID_ROWS,
            cols=config.GRID_COLS,
            max_steps=config.MAX_STEPS_PER_EPISODE,
        )
        self.agent = QLearningAgent(
            rows=self.env.rows,
            cols=self.env.cols,
            num_actions=config.NUM_ACTIONS,
            learning_rate=config.LEARNING_RATE,
            discount_factor=config.DISCOUNT_FACTOR,
            epsilon=config.EPSILON_INITIAL,
            epsilon_min=config.EPSILON_MIN,
            epsilon_decay=config.EPSILON_DECAY,
        )
        self.renderer = CampusRenderer(
            self.screen,
            rows=config.GRID_ROWS,
            cols=config.GRID_COLS,
            cell_size=config.GRID_CELL_SIZE,
        )
        self.menu = GameMenu(self.screen, on_settings_saved=self._on_settings_updated)

        # Attempt to load pre-existing trained model
        self.model_loaded = self.agent.load_model(config.Q_TABLE_SAVE_PATH)
        if self.model_loaded:
            self.agent.epsilon = 0.0

        # Phase 18: Diagnostics for missing vs corrupted models
        self.no_model_message: str = "No trained model found. Please train the AI first."
        self.no_model_detail: str = "The file 'models/q_table.npy' does not exist yet."

        # Application state management: "MENU", "TRAINING", "TRAINING_COMPLETE", "AI_PLAY", "HUMAN_PLAY"
        self.current_state = "MENU"
        self.is_running = True

        # Playback Timing
        self.ai_play_delay_ms: int = 200
        self.last_ai_step_time: int = pygame.time.get_ticks()

        # Telemetry stats dictionary
        self.episode_counter: int = 1
        self.cumulative_reward: float = 0.0
        self.step_count: int = 0
        self.last_action_name: str = "None"
        self.status_message: str = "Ready"
        self.goal_reached: bool = False
        self.paused: bool = False
        self.human_path: List[Tuple[int, int]] = []
        self.human_elapsed_time: float = 0.0
        self.human_collisions: int = 0
        self.human_hazards: int = 0
        self.last_step_reward: float = 0.0
        self.human_start_time: float = 0.0

        # ----------------------------------------------------------------------
        # Phase 8 & 14: AI Training Mode State Variables
        # ----------------------------------------------------------------------
        self.train_ep: int = 1
        self.train_total_episodes: int = config.DEFAULT_TRAINING_EPISODES
        self.train_speed: str = config.DEFAULT_TRAINING_SPEED  # "SLOW", "NORMAL", "FAST"
        self.train_records: List[Dict] = []
        self.train_ep_reward: float = 0.0
        self.train_ep_steps: int = 0
        self.train_ep_collisions: int = 0
        self.train_total_collisions: int = 0
        self.train_successes: int = 0
        self.train_outcome: str = "Starting..."
        self.train_last_step_time: int = pygame.time.get_ticks()
        self.train_summary: Dict = {}

        # ----------------------------------------------------------------------
        # Phase 11: Human vs AI Comparison State Variables
        # ----------------------------------------------------------------------
        self.comparison_human_data: Dict = {}
        self.comparison_ai_data: Dict = {}
        self.comparison_ai_path: List[Tuple[int, int]] = []
        self.comparison_ai_steps: int = 0
        self.comparison_ai_reward: float = 0.0
        self.comparison_ai_collisions: int = 0
        self.comparison_ai_start_time: float = 0.0
        self.comparison_ai_elapsed_time: float = 0.0
        self.comparison_ai_done: bool = False
        self.comparison_ai_action_name: str = "None"
        self.comparison_ai_last_step_time: int = pygame.time.get_ticks()
        self.comparison_ai_delay_ms: int = 180

        # ----------------------------------------------------------------------
        # Phase 16: Multiple Destinations & Start Locations
        # ----------------------------------------------------------------------
        self.selected_start_name: str = config.START_MAIN_GATE
        self.selected_dest_name: str = config.DEST_LIBRARY
        self.temp_selected_start: str = self.selected_start_name
        self.temp_selected_dest: str = self.selected_dest_name
        self.show_route_modal: bool = False

        # ----------------------------------------------------------------------
        # Phase 17: Visual Feedback, Collision Alerts & Pulse Animation State
        # ----------------------------------------------------------------------
        self.last_collision_time: int = 0
        self.last_collision_cell: Optional[Tuple[int, int]] = None
        self.last_collision_msg: str = ""
        self.goal_arrival_time: int = 0

        # Initialize Phase 8 Training UI Buttons & Route Modal Buttons
        self._init_training_ui_buttons()
        self._init_route_modal_buttons()

    def _init_training_ui_buttons(self) -> None:
        """Create buttons for training speed control, stopping, and completion."""
        btn_y = self.renderer.panel_y + 395
        btn_w = 104
        gap = 8

        # 3 Speed Control Buttons
        self.btn_slow = Button(
            rect=pygame.Rect(self.renderer.panel_x + 20, btn_y, btn_w, 36),
            text="Slow",
            shortcut_key="1",
            font=self.renderer.font_small,
        )
        self.btn_normal = Button(
            rect=pygame.Rect(self.renderer.panel_x + 20 + btn_w + gap, btn_y, btn_w, 36),
            text="Normal",
            shortcut_key="2",
            font=self.renderer.font_small,
        )
        self.btn_fast = Button(
            rect=pygame.Rect(self.renderer.panel_x + 20 + (btn_w + gap) * 2, btn_y, btn_w, 36),
            text="Fast",
            shortcut_key="3",
            font=self.renderer.font_small,
        )
        self.speed_buttons = [self.btn_slow, self.btn_normal, self.btn_fast]

        # Stop Training Button
        self.btn_stop = Button(
            rect=pygame.Rect(self.renderer.panel_x + 20, btn_y + 50, self.renderer.panel_width - 40, 42),
            text="Stop Training",
            shortcut_key="ESC",
            font=self.renderer.font_btn,
            base_color=(254, 242, 242),
            hover_color=(254, 226, 226),
            text_color=(220, 38, 38),
            hover_text_color=(185, 28, 28),
            border_color=(252, 165, 165),
            hover_border_color=(239, 68, 68),
        )

        # Completion Screen Action Buttons
        self.btn_complete_watch = Button(
            rect=pygame.Rect(config.SCREEN_WIDTH // 2 - 230, 620, 210, 46),
            text="Watch Trained AI",
            font=self.renderer.font_btn,
            base_color=(240, 253, 244),
            hover_color=(220, 252, 231),
            text_color=(22, 101, 52),
            hover_text_color=(21, 128, 61),
            border_color=(187, 247, 208),
        )
        self.btn_complete_menu = Button(
            rect=pygame.Rect(config.SCREEN_WIDTH // 2 + 20, 620, 210, 46),
            text="Return to Menu",
            shortcut_key="ESC",
            font=self.renderer.font_btn,
        )
        self.completion_buttons = [self.btn_complete_watch, self.btn_complete_menu]

        # Phase 9 Watch AI Action Buttons & Route Selector
        panel_w = self.renderer.panel_width - 40
        btn_y = self.renderer.panel_y + self.renderer.panel_height - 54
        route_btn_y = btn_y - 46
        btn_w_half = (panel_w - 12) // 2

        self.btn_watch_change_route = Button(
            rect=pygame.Rect(self.renderer.panel_x + 20, route_btn_y, panel_w, 38),
            text="Change Route",
            shortcut_key="C",
            font=self.renderer.font_btn,
            base_color=(238, 242, 255),
            hover_color=(224, 231, 255),
            text_color=(67, 56, 202),
            hover_text_color=(79, 70, 229),
            border_color=(199, 210, 254),
        )
        self.btn_watch_restart = Button(
            rect=pygame.Rect(self.renderer.panel_x + 20, btn_y, btn_w_half, 44),
            text="Restart",
            shortcut_key="R",
            font=self.renderer.font_btn,
            base_color=(240, 253, 244),
            hover_color=(220, 252, 231),
            text_color=(22, 101, 52),
            border_color=(187, 247, 208),
        )
        self.btn_watch_menu = Button(
            rect=pygame.Rect(self.renderer.panel_x + 20 + btn_w_half + 12, btn_y, btn_w_half, 44),
            text="Back to Menu",
            shortcut_key="ESC",
            font=self.renderer.font_btn,
        )

        # No Model Error Screen Buttons
        self.btn_no_model_train = Button(
            rect=pygame.Rect(config.SCREEN_WIDTH // 2 - 210, 430, 200, 44),
            text="Train AI Now",
            shortcut_key="1",
            font=self.renderer.font_btn,
            base_color=(240, 253, 244),
            hover_color=(220, 252, 231),
            text_color=(22, 101, 52),
            border_color=(187, 247, 208),
        )
        self.btn_no_model_menu = Button(
            rect=pygame.Rect(config.SCREEN_WIDTH // 2 + 20, 430, 200, 44),
            text="Back to Menu",
            shortcut_key="ESC",
            font=self.renderer.font_btn,
        )
        self.no_model_buttons = [self.btn_no_model_train, self.btn_no_model_menu]

        # Phase 10: Human Manual Play Action Buttons
        self.btn_human_change_route = Button(
            rect=pygame.Rect(self.renderer.panel_x + 20, route_btn_y, panel_w, 38),
            text="Change Route",
            shortcut_key="C",
            font=self.renderer.font_btn,
            base_color=(238, 242, 255),
            hover_color=(224, 231, 255),
            text_color=(67, 56, 202),
            hover_text_color=(79, 70, 229),
            border_color=(199, 210, 254),
        )
        self.btn_human_restart = Button(
            rect=pygame.Rect(self.renderer.panel_x + 20, btn_y, btn_w_half, 44),
            text="Restart",
            shortcut_key="R",
            font=self.renderer.font_btn,
            base_color=(240, 253, 244),
            hover_color=(220, 252, 231),
            text_color=(22, 101, 52),
            border_color=(187, 247, 208),
        )
        self.btn_human_menu = Button(
            rect=pygame.Rect(self.renderer.panel_x + 20 + btn_w_half + 12, btn_y, btn_w_half, 44),
            text="Back to Menu",
            shortcut_key="ESC",
            font=self.renderer.font_btn,
        )

        # Phase 10: Mission Complete Modal Action Buttons (3 Buttons)
        btn_w_3 = 210
        gap_3 = 20
        start_x_3 = 215
        btn_y_3 = 576

        self.btn_human_play_again = Button(
            rect=pygame.Rect(start_x_3, btn_y_3, btn_w_3, 44),
            text="Play Again",
            shortcut_key="R",
            font=self.renderer.font_btn,
            base_color=(240, 253, 244),
            hover_color=(220, 252, 231),
            text_color=(22, 101, 52),
            hover_text_color=(21, 128, 61),
            border_color=(187, 247, 208),
        )
        self.btn_human_watch_ai = Button(
            rect=pygame.Rect(start_x_3 + btn_w_3 + gap_3, btn_y_3, btn_w_3, 44),
            text="Compare with AI",
            shortcut_key="C",
            font=self.renderer.font_btn,
            base_color=(239, 246, 255),
            hover_color=(219, 234, 254),
            text_color=(29, 78, 216),
            hover_text_color=(30, 64, 175),
            border_color=(191, 219, 254),
        )
        self.btn_human_victory_menu = Button(
            rect=pygame.Rect(start_x_3 + (btn_w_3 + gap_3) * 2, btn_y_3, btn_w_3, 44),
            text="Back to Menu",
            shortcut_key="ESC",
            font=self.renderer.font_btn,
        )

        # Phase 11: AI Comparison Run Screen Buttons (in HUD panel)
        self.btn_comp_run_skip = Button(
            rect=pygame.Rect(self.renderer.panel_x + 20, btn_y, btn_w_half, 44),
            text="Skip to Table",
            shortcut_key="SPACE",
            font=self.renderer.font_btn,
            base_color=(239, 246, 255),
            hover_color=(219, 234, 254),
            text_color=(29, 78, 216),
            hover_text_color=(30, 64, 175),
            border_color=(191, 219, 254),
        )
        self.btn_comp_run_menu = Button(
            rect=pygame.Rect(self.renderer.panel_x + 20 + btn_w_half + 12, btn_y, btn_w_half, 44),
            text="Back to Menu",
            shortcut_key="ESC",
            font=self.renderer.font_btn,
        )

        # Phase 11: Comparison Summary Screen Buttons (4 Buttons)
        btn_w_comp = 180
        gap_comp = 16
        start_x_comp = (config.SCREEN_WIDTH - (btn_w_comp * 4 + gap_comp * 3)) // 2
        btn_y_comp = 575
        btn_h_comp = 44

        self.btn_comp_play_again = Button(
            rect=pygame.Rect(start_x_comp, btn_y_comp, btn_w_comp, btn_h_comp),
            text="Play Again",
            shortcut_key="1",
            font=self.renderer.font_btn,
            base_color=(240, 253, 244),
            hover_color=(220, 252, 231),
            text_color=(22, 101, 52),
            hover_text_color=(21, 128, 61),
            border_color=(187, 247, 208),
        )
        self.btn_comp_replay_ai = Button(
            rect=pygame.Rect(start_x_comp + btn_w_comp + gap_comp, btn_y_comp, btn_w_comp, btn_h_comp),
            text="Replay AI",
            shortcut_key="2",
            font=self.renderer.font_btn,
            base_color=(243, 232, 255),
            hover_color=(233, 213, 255),
            text_color=(126, 34, 206),
            hover_text_color=(107, 33, 168),
            border_color=(216, 180, 254),
        )
        self.btn_comp_view_map = Button(
            rect=pygame.Rect(start_x_comp + (btn_w_comp + gap_comp) * 2, btn_y_comp, btn_w_comp, btn_h_comp),
            text="View Routes",
            shortcut_key="3",
            font=self.renderer.font_btn,
            base_color=(239, 246, 255),
            hover_color=(219, 234, 254),
            text_color=(29, 78, 216),
            hover_text_color=(30, 64, 175),
            border_color=(191, 219, 254),
        )
        self.btn_comp_menu = Button(
            rect=pygame.Rect(start_x_comp + (btn_w_comp + gap_comp) * 3, btn_y_comp, btn_w_comp, btn_h_comp),
            text="Back to Menu",
            shortcut_key="ESC",
            font=self.renderer.font_btn,
        )
        self.comparison_summary_buttons = [
            self.btn_comp_play_again,
            self.btn_comp_replay_ai,
            self.btn_comp_view_map,
            self.btn_comp_menu,
        ]
        self.comparison_view_mode: str = "TABLE"

        # Comparison Map View action buttons (positioned in HUD panel)
        map_btn_w = self.renderer.panel_width - 40
        map_btn_y = self.renderer.panel_y + self.renderer.panel_height - 96
        self.btn_map_back_table = Button(
            rect=pygame.Rect(self.renderer.panel_x + 20, map_btn_y, map_btn_w, 40),
            text="Back to Table",
            shortcut_key="ESC",
            font=self.renderer.font_btn,
            base_color=(240, 253, 244),
            hover_color=(220, 252, 231),
            text_color=(22, 101, 52),
            hover_text_color=(21, 128, 61),
            border_color=(187, 247, 208),
        )
        self.btn_map_replay = Button(
            rect=pygame.Rect(self.renderer.panel_x + 20, map_btn_y + 46, map_btn_w, 40),
            text="Replay AI Run",
            shortcut_key="2",
            font=self.renderer.font_btn,
            base_color=(243, 232, 255),
            hover_color=(233, 213, 255),
            text_color=(126, 34, 206),
            hover_text_color=(107, 33, 168),
            border_color=(216, 180, 254),
        )

    def _init_route_modal_buttons(self) -> None:
        """Initialize interactive buttons for the Route & Destination Selection Modal."""
        card_w, card_h = 880, 580
        card_x = (config.SCREEN_WIDTH - card_w) // 2
        card_y = (config.SCREEN_HEIGHT - card_h) // 2
        btn_w, btn_h, gap = 155, 48, 12
        start_x = card_x + 28

        # 5 Start Location Buttons
        self.modal_start_buttons: List[Button] = []
        for i, (name, coord) in enumerate(config.START_LOCATIONS.items()):
            rect = pygame.Rect(start_x + i * (btn_w + gap), card_y + 116, btn_w, btn_h)
            btn = Button(
                rect=rect,
                text=name,
                subtext=f"({coord[0]}, {coord[1]})",
                font=self.renderer.font_small,
                center_text=True,
                base_color=(255, 255, 255),
                hover_color=(240, 253, 244),
                text_color=(30, 41, 59),
                hover_text_color=(22, 101, 52),
                border_color=(203, 213, 225),
                selected_color=(220, 252, 231),
                selected_border_color=(22, 163, 74),
                selected_text_color=(22, 101, 52),
            )
            self.modal_start_buttons.append(btn)

        # 5 Destination Buttons
        self.modal_dest_buttons: List[Button] = []
        for i, (name, coord) in enumerate(config.DESTINATIONS.items()):
            rect = pygame.Rect(start_x + i * (btn_w + gap), card_y + 226, btn_w, btn_h)
            btn = Button(
                rect=rect,
                text=name,
                subtext=f"({coord[0]}, {coord[1]})",
                font=self.renderer.font_small,
                center_text=True,
                base_color=(255, 255, 255),
                hover_color=(254, 243, 199),
                text_color=(30, 41, 59),
                hover_text_color=(180, 83, 9),
                border_color=(203, 213, 225),
                selected_color=(254, 240, 138),
                selected_border_color=(217, 119, 6),
                selected_text_color=(120, 53, 15),
            )
            self.modal_dest_buttons.append(btn)

        # Action Buttons [Confirm & Navigate] and [Cancel]
        action_w = 230
        cancel_w = 170
        gap_act = 20
        tot_act = action_w + gap_act + cancel_w
        act_x = card_x + (card_w - tot_act) // 2
        act_y = card_y + card_h - 64

        self.btn_modal_confirm = Button(
            rect=pygame.Rect(act_x, act_y, action_w, 46),
            text="Confirm & Navigate",
            shortcut_key="ENTER",
            font=self.renderer.font_btn,
            base_color=(240, 253, 244),
            hover_color=(220, 252, 231),
            text_color=(22, 101, 52),
            border_color=(187, 247, 208),
        )
        self.btn_modal_cancel = Button(
            rect=pygame.Rect(act_x + action_w + gap_act, act_y, cancel_w, 46),
            text="Cancel",
            shortcut_key="ESC",
            font=self.renderer.font_btn,
        )

    # --------------------------------------------------------------------------
    # Main Application Loop
    # --------------------------------------------------------------------------

    def run(self) -> None:
        """Main application lifecycle dispatcher."""
        while self.is_running:
            if self.current_state == "MENU":
                self._handle_menu_state()
            elif self.current_state == "TRAINING":
                self._handle_training_state()
            elif self.current_state == "TRAINING_COMPLETE":
                self._handle_training_complete_state()
            elif self.current_state == "AI_PLAY":
                self._handle_ai_play_state()
            elif self.current_state == "NO_MODEL":
                self._handle_no_model_state()
            elif self.current_state == "HUMAN_PLAY":
                self._handle_human_play_state()
            elif self.current_state == "AI_COMPARISON_RUN":
                self._handle_ai_comparison_run_state()
            elif self.current_state == "COMPARISON_SUMMARY":
                self._handle_comparison_summary_state()

        pygame.quit()
        sys.exit(0)

    def _on_settings_updated(self) -> None:
        """Dynamically reload environment, agent, and renderer when settings change."""
        self.env = CampusEnvironment(
            rows=config.GRID_ROWS,
            cols=config.GRID_COLS,
            max_steps=config.MAX_STEPS_PER_EPISODE,
        )
        self.agent = QLearningAgent(
            rows=self.env.rows,
            cols=self.env.cols,
            num_actions=config.NUM_ACTIONS,
            learning_rate=config.LEARNING_RATE,
            discount_factor=config.DISCOUNT_FACTOR,
            epsilon=config.EPSILON_INITIAL,
            epsilon_min=config.EPSILON_MIN,
            epsilon_decay=config.EPSILON_DECAY,
        )
        self.renderer = CampusRenderer(
            self.screen,
            rows=config.GRID_ROWS,
            cols=config.GRID_COLS,
            cell_size=config.GRID_CELL_SIZE,
        )
        self.train_speed = config.DEFAULT_TRAINING_SPEED
        self.train_total_episodes = config.DEFAULT_TRAINING_EPISODES
        self.model_loaded = self.agent.load_model(config.Q_TABLE_SAVE_PATH)
        if self.model_loaded:
            self.agent.epsilon = 0.0

    # --------------------------------------------------------------------------
    # Menu State Handler
    # --------------------------------------------------------------------------

    def _handle_menu_state(self) -> None:
        """Render and handle menu mode selection."""
        self.menu.draw()
        selection = self.menu.handle_input()
        self.clock.tick(config.FPS)

        if selection == 1:
            # [1] Train AI -> Enter interactive training mode
            self._start_training_mode()
        elif selection == 2:
            # [2] Watch Trained AI
            self._start_ai_mode()
        elif selection == 3:
            # [3] Play Game
            self._start_human_mode()
        elif selection == 4:
            # [4] Compare Human vs AI
            self._start_ai_comparison()
        elif selection in (6, 7, 8):
            # [8] Exit (or legacy 6, 7)
            self.is_running = False

    # --------------------------------------------------------------------------
    # Phase 8: Interactive AI Training Mode
    # --------------------------------------------------------------------------

    def _start_training_mode(self) -> None:
        """Initialize a fresh Q-table and enter interactive training mode."""
        print("\n" + "=" * 70)
        print(" [PHASE 8] INITIALIZING FRESH AI TRAINING MODE (TABULAR Q-LEARNING)")
        print("=" * 70)

        # 1. Reset Agent: fresh Q-table (all zeros) and initial epsilon = 1.0
        self.agent.reset()
        print(f"[+] Q-Table reset to zeros: {self.agent.q_table.shape}")
        print(f"[+] Epsilon restored to: {self.agent.epsilon}")

        # 2. Reset Training Counters
        self.train_ep = 1
        self.train_total_episodes = 600
        self.train_speed = "NORMAL"
        self.train_records = []
        self.train_ep_reward = 0.0
        self.train_ep_steps = 0
        self.train_ep_collisions = 0
        self.train_total_collisions = 0
        self.train_successes = 0
        self.train_outcome = "Exploring..."
        self.paused = False
        self.last_collision_time = 0
        self.last_collision_cell = None
        self.last_collision_msg = ""
        self.goal_arrival_time = 0

        # 3. Reset Environment to Start Landmark
        self.env.reset()
        self.train_last_step_time = pygame.time.get_ticks()
        self.current_state = "TRAINING"

    def _handle_training_state(self) -> None:
        """
        Execute frame-driven, non-blocking training steps while maintaining 60 FPS UI.
        Guarantees that UI events, mouse clicks, and speed buttons never freeze.
        """
        mouse_pos = pygame.mouse.get_pos()

        # 1. Event Handling (Non-blocking UI)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._finalize_training(completed=False)
                self.is_running = False
                return

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_s):
                    # Stop training early
                    self._finalize_training(completed=False)
                    return
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_1:
                    self.train_speed = "SLOW"
                elif event.key == pygame.K_2:
                    self.train_speed = "NORMAL"
                elif event.key == pygame.K_3:
                    self.train_speed = "FAST"

            # Check Button Mouse Clicks
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.btn_slow.is_clicked(mouse_pos, event):
                    self.train_speed = "SLOW"
                elif self.btn_normal.is_clicked(mouse_pos, event):
                    self.train_speed = "NORMAL"
                elif self.btn_fast.is_clicked(mouse_pos, event):
                    self.train_speed = "FAST"
                elif self.btn_stop.is_clicked(mouse_pos, event):
                    self._finalize_training(completed=False)
                    return

        # 2. Stepping Logic (Safe, cooperative time slicing)
        if not self.paused and self.train_ep <= self.train_total_episodes:
            now = pygame.time.get_ticks()

            if self.train_speed == "SLOW":
                # ~8 steps per second (125 ms interval)
                if now - self.train_last_step_time >= 125:
                    self.train_last_step_time = now
                    self._execute_single_training_step()

            elif self.train_speed == "NORMAL":
                # ~50 steps per second (20 ms interval)
                if now - self.train_last_step_time >= 20:
                    self.train_last_step_time = now
                    self._execute_single_training_step()

            elif self.train_speed == "FAST":
                # Multi-step burst per frame (hundreds of steps/sec, non-blocking)
                for _ in range(25):
                    if self.train_ep > self.train_total_episodes:
                        break
                    self._execute_single_training_step()

        # Check for natural training completion
        if self.train_ep > self.train_total_episodes:
            self._finalize_training(completed=True)
            return

        # 3. Continuous 60 FPS Render
        success_rate_so_far = (self.train_successes / max(1, self.train_ep - 1)) * 100.0
        stats = {
            "episode": self.train_ep,
            "total_episodes": self.train_total_episodes,
            "reward": self.train_ep_reward,
            "steps": self.train_ep_steps,
            "epsilon": self.agent.epsilon,
            "outcome": self.train_outcome,
            "collisions": self.train_total_collisions,
            "success_rate": success_rate_so_far,
            "paused": self.paused,
            "last_collision_time": self.last_collision_time,
            "last_collision_cell": self.last_collision_cell,
            "last_collision_msg": self.last_collision_msg,
            "goal_arrival_time": self.goal_arrival_time,
            "goal_reached": (self.train_outcome.startswith("GOAL")),
        }

        self.renderer.render_training(
            self.env,
            stats,
            self.speed_buttons,
            self.btn_stop,
            self.train_speed,
        )
        self.clock.tick(config.FPS)

    def _execute_single_training_step(self) -> None:
        """Perform a single RL step: choose action, step environment, update Q-table."""
        state = self.env.get_state()

        # Epsilon-greedy action selection
        action = self.agent.choose_action(state, training=True)

        # Environment transition
        next_state, reward, done, info = self.env.step(action)

        # Q-learning Bellman update
        self.agent.update(state, action, reward, next_state, done)

        # Update metrics
        self.train_ep_reward += reward
        self.train_ep_steps = self.env.step_count

        if not info.get("is_valid", True):
            self.train_ep_collisions += 1
            self.train_total_collisions += 1
            self.last_collision_time = pygame.time.get_ticks()
            self.last_collision_cell = next_state
            self.last_collision_msg = "Impassable Building Obstacle (-10)"

        is_goal = info.get("goal_reached", False)
        if is_goal:
            self.train_outcome = "GOAL REACHED! (+100)"
            self.goal_arrival_time = pygame.time.get_ticks()
        elif not info.get("is_valid", True):
            self.train_outcome = "Hit Obstacle / Wall (-10)"
        elif next_state in self.env.unsafe_zones:
            self.train_outcome = "Hazard Zone (-10)"
            self.last_collision_time = pygame.time.get_ticks()
            self.last_collision_cell = next_state
            self.last_collision_msg = "Hazard Zone: Mud / Slippery Path (-10)"
        else:
            self.train_outcome = f"Step: {info.get('event', 'Moved')}"

        # Handle Episode Completion
        if done:
            if is_goal:
                self.train_successes += 1

            # Log episode record
            self.train_records.append({
                "episode": self.train_ep,
                "total_reward": round(self.train_ep_reward, 2),
                "steps": self.train_ep_steps,
                "success": 1 if is_goal else 0,
                "collisions": self.train_ep_collisions,
                "epsilon": round(self.agent.epsilon, 4),
            })

            # Decay exploration rate
            self.agent.decay_epsilon()

            # Advance to next episode
            self.train_ep += 1
            self.env.reset()
            self.train_ep_reward = 0.0
            self.train_ep_steps = 0
            self.train_ep_collisions = 0

    def _finalize_training(self, completed: bool = True) -> None:
        """
        Cleanly finalize training:
        1. Persist learned Q-table
        2. Save training statistics CSV
        3. Automatically generate performance plots
        4. Switch to Completion Screen
        """
        print("\n[+] Finalizing training session...")
        df = pd.DataFrame(self.train_records)

        # 1. Save Q-table (Safeguard: avoid overwriting existing trained model with truncated short session)
        total_eps = len(df)
        should_save_model = completed or total_eps >= 50
        if not should_save_model:
            if not os.path.exists(config.Q_TABLE_SAVE_PATH) or os.path.getsize(config.Q_TABLE_SAVE_PATH) == 0:
                should_save_model = True
            else:
                try:
                    existing_q = np.load(config.Q_TABLE_SAVE_PATH)
                    if np.count_nonzero(self.agent.q_table) >= np.count_nonzero(existing_q):
                        should_save_model = True
                except Exception:
                    should_save_model = True

        if should_save_model:
            saved_model = self.agent.save_model(config.Q_TABLE_SAVE_PATH)
            print(f"[+] Q-Table saved to: {saved_model}")
        else:
            print(f"[!] Session ended early ({total_eps} eps). Preserving existing fully-trained Q-table.")

        # 2. Save CSV Metrics
        if len(df) > 0:
            os.makedirs(os.path.dirname(os.path.abspath(config.TRAINING_RESULTS_PATH)), exist_ok=True)
            df.to_csv(config.TRAINING_RESULTS_PATH, index=False)
            print(f"[+] Training metrics saved to: {config.TRAINING_RESULTS_PATH}")

            # 3. Generate All Plots
            generate_all_plots(verbose=False)
            print("[+] Performance plots automatically generated in results/plots/!")

        # 4. Compile Summary Dictionary
        total_eps = len(df)
        success_count = int(df["success"].sum()) if total_eps > 0 else 0
        success_rate = (success_count / max(1, total_eps)) * 100.0
        avg_reward = float(df["total_reward"].mean()) if total_eps > 0 else 0.0
        avg_steps = float(df["steps"].mean()) if total_eps > 0 else 0.0
        q_entries = int(np.count_nonzero(self.agent.q_table))

        self.train_summary = {
            "completed": completed,
            "episodes": total_eps,
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "collisions": self.train_total_collisions,
            "final_epsilon": float(self.agent.epsilon),
            "q_entries": q_entries,
        }

        self.current_state = "TRAINING_COMPLETE"

    def _handle_training_complete_state(self) -> None:
        """Render the training completion summary and handle user next steps."""
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
                return

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_m):
                    self.current_state = "MENU"
                    return
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE, pygame.K_1):
                    self._start_ai_mode()
                    return

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.btn_complete_watch.is_clicked(mouse_pos, event):
                    self._start_ai_mode()
                    return
                elif self.btn_complete_menu.is_clicked(mouse_pos, event):
                    self.current_state = "MENU"
                    return

        self.renderer.draw_training_complete_screen(self.train_summary, self.completion_buttons)
        self.clock.tick(config.FPS)

    # --------------------------------------------------------------------------
    # Phase 16: Route Selection Modal Handlers
    # --------------------------------------------------------------------------

    def _open_route_modal(self) -> None:
        """Open the interactive Route Selection Modal."""
        self.temp_selected_start = self.selected_start_name
        self.temp_selected_dest = self.selected_dest_name
        self.show_route_modal = True

    def _apply_route_selection(self) -> None:
        """Apply selected route coordinates to environment & agent and restart navigation."""
        self.selected_start_name = self.temp_selected_start
        self.selected_dest_name = self.temp_selected_dest
        self.show_route_modal = False

        self.env.set_start(self.selected_start_name)
        self.env.set_destination(self.selected_dest_name)
        self.agent.set_destination(self.selected_dest_name)

        if self.current_state == "AI_PLAY":
            self._start_ai_mode(preserve_landmarks=True)
        elif self.current_state == "HUMAN_PLAY":
            self._start_human_mode(preserve_landmarks=True)

    def _handle_route_modal_events(self, events: List[pygame.event.Event]) -> bool:
        """
        Handle user interaction with the Route Selection Modal.
        Returns True if modal remains open, False if modal closed.
        """
        mouse_pos = pygame.mouse.get_pos()

        start_coord = config.START_LOCATIONS.get(self.temp_selected_start, (0, 0))
        dest_coord = config.DESTINATIONS.get(self.temp_selected_dest, (9, 9))
        is_same = (start_coord == dest_coord)

        for event in events:
            if event.type == pygame.QUIT:
                self.is_running = False
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.show_route_modal = False
                    return False
                elif event.key == pygame.K_RETURN and not is_same:
                    self._apply_route_selection()
                    return False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Check Start buttons
                for btn in self.modal_start_buttons:
                    if btn.is_clicked(mouse_pos, event):
                        self.temp_selected_start = btn.text
                        break

                # Check Destination buttons
                for btn in self.modal_dest_buttons:
                    if btn.is_clicked(mouse_pos, event):
                        self.temp_selected_dest = btn.text
                        break

                # Check Confirm button
                if self.btn_modal_confirm.is_clicked(mouse_pos, event) and not is_same:
                    self._apply_route_selection()
                    return False

                # Check Cancel button
                if self.btn_modal_cancel.is_clicked(mouse_pos, event):
                    self.show_route_modal = False
                    return False

        return True

    # --------------------------------------------------------------------------
    # Phase 9: Watch Trained AI Mode
    # --------------------------------------------------------------------------

    def _start_ai_mode(self, preserve_landmarks: bool = True) -> None:
        """
        Configure and launch Watch Trained AI mode.
        1. Checks for trained model file.
        2. Loads models/q_table.npy or models/q_table_multi.npy.
        3. Sets epsilon = 0.
        4. Resets environment & route tracking.
        """
        model_path = config.Q_TABLE_SAVE_PATH
        multi_path = getattr(config, "Q_TABLE_MULTI_SAVE_PATH", "")

        # 1. Verify model file exists
        if not os.path.exists(model_path) and not os.path.exists(multi_path):
            self.no_model_message = "No trained model found. Please train the AI first."
            self.no_model_detail = f"The file '{os.path.basename(model_path)}' does not exist yet."
            try:
                from utils.logger import log_warning
                log_warning(f"Watch AI mode selected but no model found at '{model_path}'.")
            except ImportError:
                pass
            self.current_state = "NO_MODEL"
            return

        # 2. Load model into agent
        loaded = self.agent.load_model(model_path)
        if not loaded:
            self.no_model_message = "Trained model file could not be loaded."
            self.no_model_detail = getattr(
                self.agent, "last_load_error_msg", f"Failed to load weights from '{model_path}'."
            )
            try:
                from utils.logger import log_error
                log_error(f"Watch AI: {self.no_model_detail}")
            except ImportError:
                pass
            self.current_state = "NO_MODEL"
            return

        # Synchronize active destination with agent
        self.agent.set_destination(self.selected_dest_name)

        # 3. Freeze exploration (Pure Exploitation)
        self.agent.epsilon = 0.0

        # 4. Reset environment & route history
        self.env.reset()
        self.ai_path: List[Tuple[int, int]] = [self.env.agent_pos]
        self.ai_step_reward: float = 0.0
        self.ai_total_reward: float = 0.0
        self.ai_steps: int = 0
        self.ai_last_action: str = "None"
        self.ai_route_complete: bool = False
        self.ai_status: str = "Navigating Optimal Route"
        self.last_ai_step_time = pygame.time.get_ticks()
        self.paused = False
        self.last_collision_time = 0
        self.last_collision_cell = None
        self.last_collision_msg = ""
        self.goal_arrival_time = 0
        self.current_state = "AI_PLAY"

    def _handle_ai_play_state(self) -> None:
        """Step the agent autonomously using greedy policy and render route."""
        mouse_pos = pygame.mouse.get_pos()
        events = pygame.event.get()

        if self.show_route_modal:
            still_open = self._handle_route_modal_events(events)
            if not self.is_running:
                return

            # Render base view under modal
            stats = {
                "current_state": self.env.agent_pos,
                "action": self.ai_last_action,
                "reward": self.ai_step_reward,
                "total_reward": self.ai_total_reward,
                "steps": self.ai_steps,
                "status": self.ai_status,
                "goal_reached": self.ai_route_complete,
                "distance": self.env.get_distance_to_goal(),
                "start_name": self.selected_start_name,
                "destination_name": self.selected_dest_name,
                "last_collision_time": self.last_collision_time,
                "last_collision_cell": self.last_collision_cell,
                "last_collision_msg": self.last_collision_msg,
                "goal_arrival_time": self.goal_arrival_time,
            }
            self.renderer.render_watch_ai(
                self.env,
                stats,
                self.ai_path,
                self.ai_route_complete,
                self.btn_watch_restart,
                self.btn_watch_menu,
                route_btn=self.btn_watch_change_route,
            )

            # Draw Route Modal over it
            if still_open and self.show_route_modal:
                start_c = config.START_LOCATIONS.get(self.temp_selected_start, (0, 0))
                dest_c = config.DESTINATIONS.get(self.temp_selected_dest, (9, 9))
                man_dist = abs(start_c[0] - dest_c[0]) + abs(start_c[1] - dest_c[1])
                dest_id = self.agent.dest_names.index(self.temp_selected_dest) if self.temp_selected_dest in self.agent.dest_names else 0
                is_same = (start_c == dest_c)

                self.renderer.draw_route_selection_modal(
                    start_buttons=self.modal_start_buttons,
                    dest_buttons=self.modal_dest_buttons,
                    confirm_btn=self.btn_modal_confirm,
                    cancel_btn=self.btn_modal_cancel,
                    selected_start_name=self.temp_selected_start,
                    selected_dest_name=self.temp_selected_dest,
                    dest_id=dest_id,
                    manhattan_dist=man_dist,
                    is_same_location=is_same,
                )
                pygame.display.flip()

            self.clock.tick(config.FPS)
            return

        for event in events:
            if event.type == pygame.QUIT:
                self.is_running = False
                return

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_m):
                    self.current_state = "MENU"
                    return
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    self.ai_status = "Paused" if self.paused else ("DESTINATION REACHED!" if self.ai_route_complete else "Navigating Route")
                elif event.key == pygame.K_r:
                    self._start_ai_mode()
                    return
                elif event.key == pygame.K_c:
                    self._open_route_modal()
                    return
                elif event.key == pygame.K_2:
                    self._start_human_mode()
                    return
                elif event.key == pygame.K_1:
                    self._start_training_mode()
                    return

            # Check Button Mouse Clicks
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.btn_watch_change_route.is_clicked(mouse_pos, event):
                    self._open_route_modal()
                    return
                elif self.btn_watch_restart.is_clicked(mouse_pos, event):
                    self._start_ai_mode()
                    return
                elif self.btn_watch_menu.is_clicked(mouse_pos, event):
                    self.current_state = "MENU"
                    return

        # Step agent along optimal path (Human-viewable speed ~200ms per step)
        now = pygame.time.get_ticks()
        if not self.paused and not self.ai_route_complete:
            if now - self.last_ai_step_time >= self.ai_play_delay_ms:
                self.last_ai_step_time = now

                current_state = self.env.get_state()

                # Determine best action from Q-table (epsilon = 0)
                action = self.agent.get_best_action(current_state)
                next_state, reward, done, info = self.env.step(action)

                self.ai_last_action = config.ACTIONS.get(action, "None")
                self.ai_step_reward = reward
                self.ai_total_reward += reward
                self.ai_steps = self.env.step_count
                self.ai_path.append(next_state)

                if not info.get("is_valid", True):
                    self.last_collision_time = pygame.time.get_ticks()
                    self.last_collision_cell = next_state
                    self.last_collision_msg = "Impassable Building Obstacle (-10)"
                elif next_state in self.env.unsafe_zones:
                    self.last_collision_time = pygame.time.get_ticks()
                    self.last_collision_cell = next_state
                    self.last_collision_msg = "Hazard Zone: Mud / Slippery Path (-10)"

                if info.get("goal_reached", False):
                    self.ai_route_complete = True
                    self.ai_status = "DESTINATION REACHED! (+100)"
                    self.goal_arrival_time = pygame.time.get_ticks()
                elif done:
                    self.ai_route_complete = True
                    self.ai_status = "Step Limit Exceeded"
                else:
                    self.ai_status = "Navigating Optimal Route"

        # Continuous 60 FPS Render
        stats = {
            "current_state": self.env.agent_pos,
            "action": self.ai_last_action,
            "reward": self.ai_step_reward,
            "total_reward": self.ai_total_reward,
            "steps": self.ai_steps,
            "status": self.ai_status,
            "goal_reached": self.ai_route_complete,
            "distance": self.env.get_distance_to_goal(),
            "start_name": self.selected_start_name,
            "destination_name": self.selected_dest_name,
            "last_collision_time": self.last_collision_time,
            "last_collision_cell": self.last_collision_cell,
            "last_collision_msg": self.last_collision_msg,
            "goal_arrival_time": self.goal_arrival_time,
        }

        # Render Watch Mode with complete route path line
        self.renderer.render_watch_ai(
            self.env,
            stats,
            self.ai_path,
            self.ai_route_complete,
            self.btn_watch_restart,
            self.btn_watch_menu,
            route_btn=self.btn_watch_change_route,
        )
        self.clock.tick(config.FPS)

    def _handle_no_model_state(self) -> None:
        """Display error screen when user selects Watch AI without a trained model."""
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
                return

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_m):
                    self.current_state = "MENU"
                    return
                elif event.key in (pygame.K_RETURN, pygame.K_1, pygame.K_t):
                    self._start_training_mode()
                    return

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.btn_no_model_train.is_clicked(mouse_pos, event):
                    self._start_training_mode()
                    return
                elif self.btn_no_model_menu.is_clicked(mouse_pos, event):
                    self.current_state = "MENU"
                    return

        self.renderer.draw_no_model_screen(
            self.no_model_buttons,
            error_message=self.no_model_message,
            error_detail=self.no_model_detail,
        )
        self.clock.tick(config.FPS)

    # --------------------------------------------------------------------------
    # Phase 10: Manual Human Navigation Mode
    # --------------------------------------------------------------------------

    def _start_human_mode(self, preserve_landmarks: bool = True) -> None:
        """Configure and launch Manual Human Navigation mode."""
        self.env.reset()
        self.human_path: List[Tuple[int, int]] = [self.env.agent_pos]
        self.cumulative_reward = 0.0
        self.last_step_reward = 0.0
        self.step_count = 0
        self.human_collisions = 0
        self.human_hazards = 0
        self.last_action_name = "None"
        self.goal_reached = False
        self.status_message = "Use WASD or Arrow Keys"
        self.human_start_time = time.time()
        self.human_elapsed_time = 0.0
        self.last_collision_time = 0
        self.last_collision_cell = None
        self.last_collision_msg = ""
        self.goal_arrival_time = 0
        self.current_state = "HUMAN_PLAY"

    def _handle_human_play_state(self) -> None:
        """Allow manual keyboard navigation with responsive buttons and victory modal."""
        mouse_pos = pygame.mouse.get_pos()
        events = pygame.event.get()
        action_to_take: Optional[int] = None

        if self.show_route_modal:
            still_open = self._handle_route_modal_events(events)
            if not self.is_running:
                return

            time_str = f"{self.human_elapsed_time:.1f}s"
            stats = {
                "current_state": self.env.agent_pos,
                "action": self.last_action_name,
                "reward": self.last_step_reward,
                "total_reward": self.cumulative_reward,
                "steps": self.step_count,
                "time_str": time_str,
                "elapsed_seconds": self.human_elapsed_time,
                "collisions": self.human_collisions,
                "hazards": self.human_hazards,
                "distance": self.env.get_distance_to_goal(),
                "status": self.status_message,
                "goal_reached": self.goal_reached,
                "start_name": self.selected_start_name,
                "destination_name": self.selected_dest_name,
                "last_collision_time": self.last_collision_time,
                "last_collision_cell": self.last_collision_cell,
                "last_collision_msg": self.last_collision_msg,
                "goal_arrival_time": self.goal_arrival_time,
            }
            self.renderer.render_human_play(
                self.env,
                stats,
                self.human_path,
                self.btn_human_restart,
                self.btn_human_menu,
                goal_reached=self.goal_reached,
                play_again_btn=self.btn_human_play_again,
                watch_ai_btn=self.btn_human_watch_ai,
                victory_menu_btn=self.btn_human_victory_menu,
                route_btn=self.btn_human_change_route,
            )

            if still_open and self.show_route_modal:
                start_c = config.START_LOCATIONS.get(self.temp_selected_start, (0, 0))
                dest_c = config.DESTINATIONS.get(self.temp_selected_dest, (9, 9))
                man_dist = abs(start_c[0] - dest_c[0]) + abs(start_c[1] - dest_c[1])
                dest_id = self.agent.dest_names.index(self.temp_selected_dest) if self.temp_selected_dest in self.agent.dest_names else 0
                is_same = (start_c == dest_c)

                self.renderer.draw_route_selection_modal(
                    start_buttons=self.modal_start_buttons,
                    dest_buttons=self.modal_dest_buttons,
                    confirm_btn=self.btn_modal_confirm,
                    cancel_btn=self.btn_modal_cancel,
                    selected_start_name=self.temp_selected_start,
                    selected_dest_name=self.temp_selected_dest,
                    dest_id=dest_id,
                    manhattan_dist=man_dist,
                    is_same_location=is_same,
                )
                pygame.display.flip()

            self.clock.tick(config.FPS)
            return

        # Update elapsed time if still navigating
        if not self.goal_reached:
            self.human_elapsed_time = time.time() - self.human_start_time

        for event in events:
            if event.type == pygame.QUIT:
                self.is_running = False
                return

            if event.type == pygame.KEYDOWN:
                if self.goal_reached:
                    if event.key in (pygame.K_r, pygame.K_1):
                        self._start_human_mode()
                        return
                    elif event.key in (pygame.K_c, pygame.K_w, pygame.K_2, pygame.K_SPACE, pygame.K_RETURN):
                        self._start_ai_comparison()
                        return
                    elif event.key in (pygame.K_ESCAPE, pygame.K_m, pygame.K_3):
                        self.current_state = "MENU"
                        return
                else:
                    if event.key in (pygame.K_ESCAPE, pygame.K_m):
                        self.current_state = "MENU"
                        return
                    elif event.key == pygame.K_r:
                        self._start_human_mode()
                        return
                    elif event.key == pygame.K_c:
                        self._open_route_modal()
                        return
                    # Responsive movement controls (W/UP, S/DOWN, A/LEFT, D/RIGHT)
                    elif event.key in (pygame.K_w, pygame.K_UP):
                        action_to_take = config.ACTION_UP
                    elif event.key in (pygame.K_s, pygame.K_DOWN):
                        action_to_take = config.ACTION_DOWN
                    elif event.key in (pygame.K_a, pygame.K_LEFT):
                        action_to_take = config.ACTION_LEFT
                    elif event.key in (pygame.K_d, pygame.K_RIGHT):
                        action_to_take = config.ACTION_RIGHT

            # Mouse Button Clicks
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.goal_reached:
                    if self.btn_human_play_again.is_clicked(mouse_pos, event):
                        self._start_human_mode()
                        return
                    elif self.btn_human_watch_ai.is_clicked(mouse_pos, event):
                        self._start_ai_comparison()
                        return
                    elif self.btn_human_victory_menu.is_clicked(mouse_pos, event):
                        self.current_state = "MENU"
                        return
                else:
                    if self.btn_human_change_route.is_clicked(mouse_pos, event):
                        self._open_route_modal()
                        return
                    elif self.btn_human_restart.is_clicked(mouse_pos, event):
                        self._start_human_mode()
                        return
                    elif self.btn_human_menu.is_clicked(mouse_pos, event):
                        self.current_state = "MENU"
                        return

        # Execute Manual Step if playing
        if action_to_take is not None and not self.goal_reached:
            next_state, reward, done, info = self.env.step(action_to_take)
            self.last_action_name = config.ACTIONS.get(action_to_take, "None")
            self.last_step_reward = reward
            self.cumulative_reward += reward
            self.step_count = self.env.step_count
            self.goal_reached = info.get("goal_reached", False)

            # Record path history
            self.human_path.append(next_state)

            # Fault & Event Tracking
            if not info.get("is_valid", True):
                self.human_collisions += 1
                self.status_message = "Hit Obstacle / Wall (-10)"
                self.last_collision_time = pygame.time.get_ticks()
                self.last_collision_cell = next_state
                self.last_collision_msg = "Impassable Building Obstacle (-10)"
            elif next_state in self.env.unsafe_zones:
                self.human_hazards += 1
                self.status_message = "Hazard Zone (-10)"
                self.last_collision_time = pygame.time.get_ticks()
                self.last_collision_cell = next_state
                self.last_collision_msg = "Hazard Zone: Mud / Slippery Path (-10)"
            elif self.goal_reached:
                self.status_message = "MISSION COMPLETE! (+100)"
                self.human_elapsed_time = time.time() - self.human_start_time
                self.goal_arrival_time = pygame.time.get_ticks()
            else:
                self.status_message = f"Step: {info.get('event', 'Moved')}"

        # Format elapsed duration string
        time_str = f"{self.human_elapsed_time:.1f}s"

        stats = {
            "current_state": self.env.agent_pos,
            "action": self.last_action_name,
            "reward": self.last_step_reward,
            "total_reward": self.cumulative_reward,
            "steps": self.step_count,
            "time_str": time_str,
            "elapsed_seconds": self.human_elapsed_time,
            "collisions": self.human_collisions,
            "hazards": self.human_hazards,
            "distance": self.env.get_distance_to_goal(),
            "status": self.status_message,
            "goal_reached": self.goal_reached,
            "start_name": self.selected_start_name,
            "destination_name": self.selected_dest_name,
            "last_collision_time": self.last_collision_time,
            "last_collision_cell": self.last_collision_cell,
            "last_collision_msg": self.last_collision_msg,
            "goal_arrival_time": self.goal_arrival_time,
        }

        self.renderer.render_human_play(
            self.env,
            stats,
            self.human_path,
            self.btn_human_restart,
            self.btn_human_menu,
            goal_reached=self.goal_reached,
            play_again_btn=self.btn_human_play_again,
            watch_ai_btn=self.btn_human_watch_ai,
            victory_menu_btn=self.btn_human_victory_menu,
            route_btn=self.btn_human_change_route,
        )
        self.clock.tick(config.FPS)

    # --------------------------------------------------------------------------
    # Phase 11: Human vs AI Comparison Mode
    # --------------------------------------------------------------------------

    def _start_ai_comparison(self) -> None:
        """
        Record final human navigation metrics and launch the trained AI on the identical
        campus environment to measure and display comparative performance.
        """
        # Ensure trained model file exists before proceeding
        if not os.path.exists(config.Q_TABLE_SAVE_PATH) and not os.path.exists(getattr(config, "Q_TABLE_MULTI_SAVE_PATH", "")):
            self.no_model_message = "No trained model found. Please train the AI first."
            self.no_model_detail = "AI comparison requires a trained model ('models/q_table.npy')."
            try:
                from utils.logger import log_warning
                log_warning("AI comparison requested without a trained model file.")
            except ImportError:
                pass
            self.current_state = "NO_MODEL"
            return

        # Load Q-table, synchronize destination policy, and set greedy exploitation (epsilon = 0.0)
        loaded = self.agent.load_model()
        if not loaded:
            self.no_model_message = "Trained model file could not be loaded."
            self.no_model_detail = getattr(
                self.agent, "last_load_error_msg", "Model weights file is corrupted or incompatible."
            )
            try:
                from utils.logger import log_error
                log_error(f"AI comparison: {self.no_model_detail}")
            except ImportError:
                pass
            self.current_state = "NO_MODEL"
            return

        # Ensure start and destination coordinates are synchronized
        start_coord = config.START_LOCATIONS.get(self.selected_start_name, config.START_POSITION)
        dest_coord = config.DESTINATIONS.get(self.selected_dest_name, config.GOAL_POSITION)
        self.env.set_start(start_coord)
        self.env.set_destination(dest_coord)
        self.agent.set_destination(self.selected_dest_name)
        self.agent.epsilon = 0.0

        # Snapshot human navigation results if played, or establish baseline
        if self.step_count > 0:
            self.comparison_human_data = {
                "steps": self.step_count,
                "reward": round(self.cumulative_reward, 1),
                "time": f"{self.human_elapsed_time:.1f}s",
                "route_length": len(self.human_path),
                "success": "Yes" if self.goal_reached else "No",
                "collisions": self.human_collisions,
            }
        else:
            if not self.comparison_human_data:
                self.comparison_human_data = {
                    "steps": 32,
                    "reward": 61.0,
                    "time": "14.2s",
                    "route_length": 33,
                    "success": "Yes",
                    "collisions": 1,
                }
            # Provide realistic baseline human trajectory if none recorded yet
            if not self.human_path or len(self.human_path) < 2:
                self.human_path = [
                    (0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (1, 4), (1, 3),
                    (2, 3), (2, 4), (3, 4), (3, 3), (4, 3), (4, 2), (4, 3),
                    (5, 3), (5, 4), (6, 4), (6, 5), (6, 6), (7, 6), (7, 5),
                    (7, 4), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (9, 9)
                ]

        # Reset identical campus environment for AI evaluation run
        self.env.reset()
        self.comparison_ai_path = [self.env.agent_pos]
        self.comparison_ai_steps = 0
        self.comparison_ai_reward = 0.0
        self.comparison_ai_collisions = 0
        self.comparison_ai_done = False
        self.comparison_ai_action_name = "None"
        self.comparison_ai_start_time = time.time()
        self.comparison_ai_elapsed_time = 0.0
        self.comparison_ai_last_step_time = pygame.time.get_ticks()
        self.last_collision_time = 0
        self.last_collision_cell = None
        self.last_collision_msg = ""
        self.goal_arrival_time = 0
        self.comparison_view_mode = "TABLE"
        self.btn_comp_run_skip.text = "Skip to Table"

        self.current_state = "AI_COMPARISON_RUN"

    def _step_ai_comparison(self) -> None:
        """Execute a single step for the AI in comparison run mode."""
        if self.comparison_ai_done:
            return

        curr_state = self.env.get_state()
        action = self.agent.get_best_action(curr_state)
        next_state, reward, done, info = self.env.step(action)

        self.comparison_ai_action_name = config.ACTIONS.get(action, "None")
        self.comparison_ai_reward += reward
        self.comparison_ai_steps = self.env.step_count
        self.comparison_ai_path.append(next_state)

        if not info.get("is_valid", True):
            self.comparison_ai_collisions += 1
            self.last_collision_time = pygame.time.get_ticks()
            self.last_collision_cell = next_state
            self.last_collision_msg = "Impassable Building Obstacle (-10)"
        elif next_state in self.env.unsafe_zones:
            self.last_collision_time = pygame.time.get_ticks()
            self.last_collision_cell = next_state
            self.last_collision_msg = "Hazard Zone: Mud / Slippery Path (-10)"

        is_goal = info.get("goal_reached", False) or (self.env.agent_pos == self.env.goal_pos)
        if is_goal:
            self.goal_arrival_time = pygame.time.get_ticks()

        if done or is_goal or self.comparison_ai_steps >= config.MAX_STEPS_PER_EPISODE:
            self.comparison_ai_done = True
            self.comparison_ai_elapsed_time = time.time() - self.comparison_ai_start_time
            self.btn_comp_run_skip.text = "View Comparison Table"

    def _fast_forward_ai_comparison(self) -> None:
        """Instantly simulate the AI route to the end and transition to comparison table."""
        while not self.comparison_ai_done:
            self._step_ai_comparison()
        self._finalize_ai_comparison()

    def _finalize_ai_comparison(self) -> None:
        """Finalize AI evaluation metrics and transition to the side-by-side table."""
        self.comparison_ai_data = {
            "steps": self.comparison_ai_steps,
            "reward": round(self.comparison_ai_reward, 1),
            "time": f"{self.comparison_ai_elapsed_time:.1f}s",
            "route_length": len(self.comparison_ai_path),
            "success": "Yes" if self.env.agent_pos == self.env.goal_pos else "No",
            "collisions": self.comparison_ai_collisions,
        }
        self.comparison_view_mode = "TABLE"
        self.current_state = "COMPARISON_SUMMARY"

    def _handle_ai_comparison_run_state(self) -> None:
        """Handle events, stepping, and rendering for the live AI comparison run."""
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
                return

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_m):
                    self.current_state = "MENU"
                    return
                elif event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_s):
                    if self.comparison_ai_done:
                        self._finalize_ai_comparison()
                    else:
                        self._fast_forward_ai_comparison()
                    return

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.btn_comp_run_skip.is_clicked(mouse_pos, event):
                    if self.comparison_ai_done:
                        self._finalize_ai_comparison()
                    else:
                        self._fast_forward_ai_comparison()
                    return
                elif self.btn_comp_run_menu.is_clicked(mouse_pos, event):
                    self.current_state = "MENU"
                    return

        # Advance AI step based on timer
        now = pygame.time.get_ticks()
        if not self.comparison_ai_done:
            self.comparison_ai_elapsed_time = time.time() - self.comparison_ai_start_time
            if now - self.comparison_ai_last_step_time >= self.comparison_ai_delay_ms:
                self._step_ai_comparison()
                self.comparison_ai_last_step_time = now
        else:
            # When done, allow arrival frame to show, then auto-advance after 1.4s
            if self.goal_arrival_time > 0 and (now - self.goal_arrival_time >= 1400):
                self._finalize_ai_comparison()
                return

        # If step finalized and transitioned to summary, exit handler
        if self.current_state != "AI_COMPARISON_RUN":
            return

        # Prepare stats dictionary for live HUD
        stats = {
            "current_state": self.env.agent_pos,
            "action": self.comparison_ai_action_name,
            "reward": self.comparison_ai_reward,
            "steps": self.comparison_ai_steps,
            "human_steps": self.comparison_human_data.get("steps", 0),
            "human_reward": self.comparison_human_data.get("reward", 0.0),
            "time_str": f"{self.comparison_ai_elapsed_time:.1f}s",
            "distance": self.env.get_distance_to_goal(),
            "goal_reached": (self.env.agent_pos == self.env.goal_pos),
            "last_collision_time": self.last_collision_time,
            "last_collision_cell": self.last_collision_cell,
            "last_collision_msg": self.last_collision_msg,
            "goal_arrival_time": self.goal_arrival_time,
            "start_name": self.selected_start_name,
            "destination_name": self.selected_dest_name,
        }

        self.renderer.render_ai_comparison_run(
            self.env,
            stats,
            self.human_path,
            self.comparison_ai_path,
            self.btn_comp_run_skip,
            self.btn_comp_run_menu,
        )
        self.clock.tick(config.FPS)

    def _handle_comparison_summary_state(self) -> None:
        """Render and handle actions on the side-by-side Human vs AI comparison screen."""
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
                return

            if event.type == pygame.KEYDOWN:
                if self.comparison_view_mode == "MAP":
                    if event.key in (pygame.K_ESCAPE, pygame.K_m, pygame.K_t, pygame.K_3, pygame.K_v, pygame.K_SPACE, pygame.K_RETURN):
                        self.comparison_view_mode = "TABLE"
                        return
                    elif event.key in (pygame.K_2, pygame.K_a, pygame.K_r):
                        self._start_ai_comparison()
                        return
                    elif event.key in (pygame.K_1, pygame.K_p):
                        self._start_human_mode()
                        return
                else:
                    if event.key in (pygame.K_1, pygame.K_p, pygame.K_r):
                        self._start_human_mode()
                        return
                    elif event.key in (pygame.K_2, pygame.K_a):
                        self._start_ai_comparison()
                        return
                    elif event.key in (pygame.K_3, pygame.K_v, pygame.K_m, pygame.K_SPACE):
                        self.comparison_view_mode = "MAP"
                        return
                    elif event.key in (pygame.K_ESCAPE, pygame.K_4):
                        self.current_state = "MENU"
                        return

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.comparison_view_mode == "MAP":
                    if self.btn_map_back_table.is_clicked(mouse_pos, event):
                        self.comparison_view_mode = "TABLE"
                        return
                    elif self.btn_map_replay.is_clicked(mouse_pos, event):
                        self._start_ai_comparison()
                        return
                else:
                    if self.btn_comp_play_again.is_clicked(mouse_pos, event):
                        self._start_human_mode()
                        return
                    elif self.btn_comp_replay_ai.is_clicked(mouse_pos, event):
                        self._start_ai_comparison()
                        return
                    elif self.btn_comp_view_map.is_clicked(mouse_pos, event):
                        self.comparison_view_mode = "MAP"
                        return
                    elif self.btn_comp_menu.is_clicked(mouse_pos, event):
                        self.current_state = "MENU"
                        return

        if self.comparison_view_mode == "MAP":
            stats = {
                "current_state": self.env.agent_pos,
                "action": "Completed",
                "reward": self.comparison_ai_reward,
                "steps": self.comparison_ai_steps,
                "human_steps": self.comparison_human_data.get("steps", 0),
                "human_reward": self.comparison_human_data.get("reward", 0.0),
                "human_collisions": self.comparison_human_data.get("collisions", 0),
                "ai_collisions": self.comparison_ai_collisions,
                "time_str": self.comparison_ai_data.get("time", "0.0s"),
                "human_time_str": self.comparison_human_data.get("time", "0.0s"),
                "distance": 0,
                "goal_reached": True,
                "start_name": self.selected_start_name,
                "destination_name": self.selected_dest_name,
            }
            self.renderer.draw_comparison_map_screen(
                self.env,
                stats,
                self.human_path,
                self.comparison_ai_path,
                self.btn_map_back_table,
                self.btn_map_replay,
            )
        else:
            self.renderer.draw_comparison_screen(
                self.comparison_human_data,
                self.comparison_ai_data,
                self.comparison_summary_buttons,
                start_name=self.selected_start_name,
                dest_name=self.selected_dest_name,
            )
        self.clock.tick(config.FPS)


if __name__ == "__main__":
    game = CampusNavigatorGame()
    game.run()
