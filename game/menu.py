"""
Campus Navigator: Educational Game Menu (Phase 7)
=================================================
Module: game/menu.py

Provides a professional, college-presentation-ready interface with:
- Reusable Button components supporting mouse hover & keyboard navigation.
- 6 Primary Menu Options:
    [1] Train AI (Run Q-Learning training loop)
    [2] Watch Trained AI (Autonomous pathfinding with learned policy)
    [3] Play Game (Manual keyboard campus navigation)
    [4] Training Results (Performance statistics & convergence metrics)
    [5] How RL Works (Educational viva theory & Bellman equations)
    [6] Exit (Close application)
- Mouse click & hover interaction + keyboard shortcuts (1-6, Arrows, ENTER, ESC).
- Built-in educational modal screens for [4] Results and [5] RL Theory Guide.
"""

import json
import os
import subprocess
import sys
from typing import Callable, Dict, List, Optional, Tuple
import pandas as pd
import pygame

# Ensure project root is accessible when running standalone
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import config
from visualization.plots import generate_all_plots
from utils.logger import log_error, log_warning, log_info


# ==============================================================================
# Reusable UI Button Component
# ==============================================================================

class Button:
    """A clean, interactive rectangular UI button with hover & click feedback."""

    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        shortcut_key: str = "",
        font: Optional[pygame.font.Font] = None,
        base_color: Tuple[int, int, int] = (255, 255, 255),
        hover_color: Tuple[int, int, int] = (239, 246, 255),
        text_color: Tuple[int, int, int] = (30, 41, 59),
        hover_text_color: Tuple[int, int, int] = (37, 99, 235),
        border_color: Tuple[int, int, int] = (203, 213, 225),
        hover_border_color: Tuple[int, int, int] = (37, 99, 235),
        border_radius: int = 8,
        center_text: bool = False,
        subtext: str = "",
        selected_color: Optional[Tuple[int, int, int]] = None,
        selected_border_color: Optional[Tuple[int, int, int]] = None,
        selected_text_color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        self.rect = rect
        self.text = text
        self.shortcut_key = shortcut_key
        self.font = font or pygame.font.SysFont("Segoe UI, Arial", 16, bold=True)
        self.base_color = base_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.hover_text_color = hover_text_color
        self.border_color = border_color
        self.hover_border_color = hover_border_color
        self.border_radius = border_radius
        self.center_text = center_text
        self.subtext = subtext
        self.selected_color = selected_color
        self.selected_border_color = selected_border_color
        self.selected_text_color = selected_text_color
        self.is_hovered: bool = False
        self.is_selected: bool = False

    def update(self, mouse_pos: Tuple[int, int]) -> None:
        """Update hover state based on cursor position."""
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def draw(self, screen: pygame.Surface) -> None:
        """Render the button on screen."""
        active = self.is_hovered or self.is_selected
        
        if self.is_selected and self.selected_color is not None:
            bg_col = self.selected_color
            border_col = self.selected_border_color or self.hover_border_color
            txt_col = self.selected_text_color or self.hover_text_color
        elif active:
            bg_col = self.hover_color
            border_col = self.hover_border_color
            txt_col = self.hover_text_color
        else:
            bg_col = self.base_color
            border_col = self.border_color
            txt_col = self.text_color

        # Background fill
        pygame.draw.rect(screen, bg_col, self.rect, border_radius=self.border_radius)
        # Border
        border_w = 2 if active else 1
        pygame.draw.rect(screen, border_col, self.rect, border_w, border_radius=self.border_radius)

        if self.center_text:
            if self.subtext:
                txt_surf = self.font.render(self.text, True, txt_col)
                sub_font = pygame.font.SysFont("Segoe UI, Arial", 11, bold=True)
                sub_surf = sub_font.render(self.subtext, True, (100, 116, 139) if not active else txt_col)
                total_h = txt_surf.get_height() + sub_surf.get_height() + 2
                start_y = self.rect.centery - total_h // 2
                screen.blit(txt_surf, (self.rect.centerx - txt_surf.get_width() // 2, start_y))
                screen.blit(sub_surf, (self.rect.centerx - sub_surf.get_width() // 2, start_y + txt_surf.get_height() + 2))
            else:
                txt_surf = self.font.render(self.text, True, txt_col)
                screen.blit(txt_surf, (self.rect.centerx - txt_surf.get_width() // 2, self.rect.centery - txt_surf.get_height() // 2))
        else:
            # Main text
            prefix = " > " if active else "   "
            full_text = f"{prefix}{self.text}"
            txt_surf = self.font.render(full_text, True, txt_col)
            screen.blit(txt_surf, (self.rect.x + 16, self.rect.centery - txt_surf.get_height() // 2))

            # Shortcut key badge (right-aligned styled pill)
            if self.shortcut_key:
                badge_font = pygame.font.SysFont("Consolas, Segoe UI, Arial", 11, bold=True)
                badge_txt = badge_font.render(f"[{self.shortcut_key}]", True, (37, 99, 235) if active else (100, 116, 139))
                badge_w = badge_txt.get_width() + 10
                badge_h = 20
                badge_rect = pygame.Rect(self.rect.right - badge_w - 14, self.rect.centery - badge_h // 2, badge_w, badge_h)
                
                # Pill background and border
                pill_bg = (239, 246, 255) if active else (241, 245, 249)
                pill_border = (191, 219, 254) if active else (226, 232, 240)
                pygame.draw.rect(screen, pill_bg, badge_rect, border_radius=4)
                pygame.draw.rect(screen, pill_border, badge_rect, 1, border_radius=4)
                screen.blit(badge_txt, (badge_rect.centerx - badge_txt.get_width() // 2, badge_rect.centery - badge_txt.get_height() // 2))

    def is_clicked(self, mouse_pos: Tuple[int, int], event: pygame.event.Event) -> bool:
        """Check if mouse button clicked this button."""
        return (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
            and self.rect.collidepoint(mouse_pos)
        )


# ==============================================================================
# Main Game Menu Controller
# ==============================================================================

class GameMenu:
    """Educational Game Menu with 6 distinct options, modal dialogs, and dual input."""

    def __init__(self, screen: pygame.Surface, on_settings_saved: Optional[Callable] = None) -> None:
        """
        Initialize the Menu controller.

        Args:
            screen: Pygame display surface (1100x750).
            on_settings_saved: Optional callback invoked when settings are saved or reset.
        """
        self.screen = screen
        self.on_settings_saved: Optional[Callable] = on_settings_saved
        self.sub_screen: str = "MAIN"  # "MAIN", "RESULTS", "RL_WORKS", "SETTINGS"
        self.selected_index: int = 0

        # Initialize Fonts
        pygame.font.init()
        self.font_title = pygame.font.SysFont("Segoe UI, Arial", 34, bold=True)
        self.font_subtitle = pygame.font.SysFont("Segoe UI, Arial", 15)
        self.font_tag = pygame.font.SysFont("Segoe UI, Arial", 11, bold=True)
        self.font_btn = pygame.font.SysFont("Segoe UI, Arial", 16, bold=True)
        self.font_body = pygame.font.SysFont("Segoe UI, Arial", 14)
        self.font_body_bold = pygame.font.SysFont("Segoe UI, Arial", 14, bold=True)
        self.font_header = pygame.font.SysFont("Segoe UI, Arial", 20, bold=True)
        self.font_footer = pygame.font.SysFont("Segoe UI, Arial", 12)
        self.font_card_title = pygame.font.SysFont("Segoe UI, Arial", 11, bold=True)
        self.font_val_large = pygame.font.SysFont("Segoe UI, Arial", 21, bold=True)
        self.font_stat_sub = pygame.font.SysFont("Segoe UI, Arial", 11)
        self.font_val = pygame.font.SysFont("Segoe UI, Arial", 14, bold=True)

        # Create Buttons for the 8 Menu Options
        self.option_labels = [
            ("1. Train AI", "1", 1),
            ("2. Watch AI", "2", 2),
            ("3. Play Game", "3", 3),
            ("4. Compare Human vs AI", "4", 4),
            ("5. View Results", "5", 5),
            ("6. Learn RL", "6", 6),
            ("7. Settings", "7", 7),
            ("8. Exit", "8", 8),
        ]

        self.buttons: List[Button] = []
        self._init_buttons()

        # Back buttons for sub-screens
        self.back_button = Button(
            rect=pygame.Rect(config.SCREEN_WIDTH // 2 - 100, 660, 200, 42),
            text="Back to Menu",
            shortcut_key="ESC",
            font=self.font_btn,
        )

        # ----------------------------------------------------------------------
        # Phase 12: Training Results Dashboard State & Buttons
        # ----------------------------------------------------------------------
        self.results_scroll_pos: float = 0.0
        self.results_max_scroll: float = 440.0
        self.results_metrics: Optional[Dict] = None
        self.results_df_loaded: bool = False
        self.results_error_msg: Optional[str] = None
        self.results_cached_plots: Dict[str, Optional[pygame.Surface]] = {}
        self.results_plots_loaded: bool = False
        self.is_dragging_scrollbar: bool = False
        self.drag_start_mouse_y: int = 0
        self.drag_start_scroll_pos: float = 0.0

        btn_w_res = 230
        btn_h_res = 44
        gap_res = 24
        start_x_res = (config.SCREEN_WIDTH - (btn_w_res * 2 + gap_res)) // 2
        btn_y_res = 690

        self.btn_open_folder = Button(
            rect=pygame.Rect(start_x_res, btn_y_res, btn_w_res, btn_h_res),
            text="Open Results Folder",
            shortcut_key="O",
            font=self.font_btn,
            base_color=(240, 253, 244),
            hover_color=(220, 252, 231),
            text_color=(22, 101, 52),
            hover_text_color=(21, 128, 61),
            border_color=(187, 247, 208),
            hover_border_color=(34, 197, 94),
        )
        self.btn_results_back = Button(
            rect=pygame.Rect(start_x_res + btn_w_res + gap_res, btn_y_res, btn_w_res, btn_h_res),
            text="Back to Menu",
            shortcut_key="ESC",
            font=self.font_btn,
        )

        # Fallback Buttons if no results found
        self.btn_no_results_train = Button(
            rect=pygame.Rect(config.SCREEN_WIDTH // 2 - 220, 440, 200, 44),
            text="Train AI Now",
            shortcut_key="1",
            font=self.font_btn,
            base_color=(240, 253, 244),
            hover_color=(220, 252, 231),
            text_color=(22, 101, 52),
            hover_text_color=(21, 128, 61),
            border_color=(187, 247, 208),
        )
        self.btn_no_results_back = Button(
            rect=pygame.Rect(config.SCREEN_WIDTH // 2 + 20, 440, 200, 44),
            text="Back to Menu",
            shortcut_key="ESC",
            font=self.font_btn,
        )

        # ----------------------------------------------------------------------
        # Phase 13: "How RL Works" Screen State & Buttons
        # ----------------------------------------------------------------------
        self.rl_scroll_pos: float = 0.0
        self.rl_max_scroll: float = 480.0
        self.is_dragging_rl_scrollbar: bool = False

        self.font_formula = pygame.font.SysFont("Consolas, Courier New, monospace", 16, bold=True)
        self.font_concept_num = pygame.font.SysFont("Segoe UI, Arial", 10, bold=True)

        self.btn_rl_back = Button(
            rect=pygame.Rect(config.SCREEN_WIDTH // 2 - 110, 690, 220, 44),
            text="Back to Menu",
            shortcut_key="ESC",
            font=self.font_btn,
        )

        # ----------------------------------------------------------------------
        # Phase 14: Settings State, Inputs & Buttons
        # ----------------------------------------------------------------------
        self.settings_data: Dict = config.load_config()
        self.settings_active_field: Optional[str] = None
        self.settings_text_buffer: str = ""
        self.settings_status: Tuple[str, str] = ("normal", "Adjust hyperparameters and simulation parameters below.")
        self.settings_rects: Dict = {}

        btn_w_set = 210
        btn_h_set = 44
        gap_set = 24
        start_x_set = (config.SCREEN_WIDTH - (btn_w_set * 3 + gap_set * 2)) // 2
        btn_y_set = 680

        self.btn_save_settings = Button(
            rect=pygame.Rect(start_x_set, btn_y_set, btn_w_set, btn_h_set),
            text="Save Settings",
            shortcut_key="S",
            font=self.font_btn,
            base_color=(240, 253, 244),
            hover_color=(220, 252, 231),
            text_color=(22, 101, 52),
            hover_text_color=(21, 128, 61),
            border_color=(187, 247, 208),
            hover_border_color=(34, 197, 94),
        )
        self.btn_reset_settings = Button(
            rect=pygame.Rect(start_x_set + btn_w_set + gap_set, btn_y_set, btn_w_set, btn_h_set),
            text="Reset Defaults",
            shortcut_key="R",
            font=self.font_btn,
            base_color=(254, 243, 199),
            hover_color=(253, 230, 138),
            text_color=(146, 64, 14),
            hover_text_color=(180, 83, 9),
            border_color=(253, 230, 138),
            hover_border_color=(245, 158, 11),
        )
        self.btn_back_settings = Button(
            rect=pygame.Rect(start_x_set + (btn_w_set + gap_set) * 2, btn_y_set, btn_w_set, btn_h_set),
            text="Back to Menu",
            shortcut_key="ESC",
            font=self.font_btn,
        )

    def _init_buttons(self) -> None:
        """Layout the 8 menu buttons centered horizontally."""
        container_w = 600
        container_x = (config.SCREEN_WIDTH - container_w) // 2
        start_y = 168
        btn_h = 44
        gap = 10

        self.buttons.clear()
        for idx, (label, key, code) in enumerate(self.option_labels):
            btn_rect = pygame.Rect(container_x, start_y + idx * (btn_h + gap), container_w, btn_h)
            btn = Button(
                rect=btn_rect,
                text=label,
                shortcut_key=key,
                font=self.font_btn,
            )
            self.buttons.append(btn)

    # --------------------------------------------------------------------------
    # Main Menu View
    # --------------------------------------------------------------------------

    def _draw_main_menu(self) -> None:
        """Render the primary main menu."""
        self.screen.fill(config.COLOR_BACKGROUND)

        # Header Title & Subtitle matching prompt
        title = self.font_title.render("CAMPUS NAVIGATOR", True, (15, 23, 42))
        sub1 = self.font_subtitle.render("An Intelligent Path-Finding Game", True, (51, 65, 85))
        sub2 = self.font_tag.render("Using Q-Learning Reinforcement Learning", True, (37, 99, 235))
        self.screen.blit(title, (config.SCREEN_WIDTH // 2 - title.get_width() // 2, 36))
        self.screen.blit(sub1, (config.SCREEN_WIDTH // 2 - sub1.get_width() // 2, 82))
        self.screen.blit(sub2, (config.SCREEN_WIDTH // 2 - sub2.get_width() // 2, 108))

        # Academic Badges
        badges = ["10x10 Virtual Campus", "Tabular Q-Learning", "MDP Environment", "College Project"]
        total_w = sum(self.font_tag.size(b)[0] + 24 for b in badges) + (len(badges) - 1) * 10
        cur_bx = (config.SCREEN_WIDTH - total_w) // 2

        for badge in badges:
            bw, bh = self.font_tag.size(badge)[0] + 20, 22
            brect = pygame.Rect(cur_bx, 134, bw, bh)
            pygame.draw.rect(self.screen, (226, 232, 240), brect, border_radius=11)
            pygame.draw.rect(self.screen, (203, 213, 225), brect, 1, border_radius=11)
            bsurf = self.font_tag.render(badge, True, (51, 65, 85))
            self.screen.blit(bsurf, (brect.centerx - bsurf.get_width() // 2, brect.centery - bsurf.get_height() // 2))
            cur_bx += bw + 10

        # Draw Buttons
        mouse_pos = pygame.mouse.get_pos()
        for idx, btn in enumerate(self.buttons):
            btn.is_selected = (idx == self.selected_index)
            btn.update(mouse_pos)
            btn.draw(self.screen)

        # Footer Instruction
        footer = self.font_footer.render(
            "Use UP / DOWN Arrow Keys and Press ENTER  |  Or Click with Mouse  |  Or Press [1] - [8] / [ESC]",
            True,
            (148, 163, 184),
        )
        self.screen.blit(footer, (config.SCREEN_WIDTH // 2 - footer.get_width() // 2, 695))

    # --------------------------------------------------------------------------
    # Sub-Screen: [4] Training Results Modal
    # --------------------------------------------------------------------------

    def _load_results_data(self, force_reload: bool = False) -> None:
        """
        Load training metrics from results/training_results.csv and compute
        the 7 key summary benchmarks.
        """
        if self.results_df_loaded and not force_reload:
            return

        csv_path = getattr(config, "TRAINING_RESULTS_SAVE_PATH", os.path.join(config.RESULTS_DIR, "training_results.csv"))
        if not os.path.exists(csv_path):
            self.results_metrics = None
            self.results_df_loaded = False
            self.results_error_msg = "No training results file found. Please train the AI agent first."
            log_warning(f"Results CSV not found at '{csv_path}'. Prompting user to train first.")
            return

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            self.results_metrics = None
            self.results_df_loaded = False
            self.results_error_msg = f"Failed to read training results CSV: {e}"
            log_error(f"Failed to read training results CSV '{csv_path}': {e}")
            return

        if len(df) == 0:
            self.results_metrics = None
            self.results_df_loaded = False
            self.results_error_msg = "Training results file is empty (0 episodes recorded). Please run training first."
            log_warning(f"Training results CSV '{csv_path}' is empty (0 episodes).")
            return

        total_episodes = len(df)
        recent_window = min(100, total_episodes)

        # 1. Success Rate
        sr_all = (df["success"].sum() / total_episodes) * 100.0
        sr_recent = (df["success"].tail(recent_window).sum() / recent_window) * 100.0

        # 2. Average Reward
        rew_all = float(df["total_reward"].mean())
        rew_recent = float(df["total_reward"].tail(recent_window).mean())

        # 3. Average Steps
        steps_all = float(df["steps"].mean())
        steps_recent = float(df["steps"].tail(recent_window).mean())

        # 4. Collision Rate
        coll_all = float(df["collisions"].mean())
        coll_recent = float(df["collisions"].tail(recent_window).mean())

        # 5. Best Episode
        succ_df = df[df["success"] == 1]
        if len(succ_df) > 0:
            best_row = succ_df.sort_values(by=["total_reward", "steps"], ascending=[False, True]).iloc[0]
        else:
            best_row = df.loc[df["total_reward"].idxmax()]

        best_ep = int(best_row["episode"])
        best_rew = float(best_row["total_reward"])
        best_steps = int(best_row["steps"])

        # 6. Final Epsilon
        final_eps = float(df.iloc[-1]["epsilon"])

        self.results_metrics = {
            "total_episodes": total_episodes,
            "sr_recent": sr_recent,
            "sr_all": sr_all,
            "rew_recent": rew_recent,
            "rew_all": rew_all,
            "steps_recent": steps_recent,
            "steps_all": steps_all,
            "coll_recent": coll_recent,
            "coll_all": coll_all,
            "best_ep": best_ep,
            "best_rew": best_rew,
            "best_steps": best_steps,
            "final_eps": final_eps,
        }
        self.results_df_loaded = True
        self.results_error_msg = None

    def _load_results_plots(self, force_reload: bool = False) -> None:
        """
        Load and smoothscale the 4 required Matplotlib plot images into memory.
        If any plot file is missing, automatically invoke generate_all_plots().
        """
        if self.results_plots_loaded and not force_reload:
            return

        plots_dir = os.path.join(config.RESULTS_DIR, "plots")
        required_plots = {
            "reward_vs_episode": os.path.join(plots_dir, "reward_vs_episode.png"),
            "steps_vs_episode": os.path.join(plots_dir, "steps_vs_episode.png"),
            "success_rate": os.path.join(plots_dir, "success_rate.png"),
            "collision_rate": os.path.join(plots_dir, "collision_rate.png"),
        }

        # Check if any plot is missing
        any_missing = any(not os.path.exists(path) for path in required_plots.values())
        csv_path = getattr(config, "TRAINING_RESULTS_SAVE_PATH", os.path.join(config.RESULTS_DIR, "training_results.csv"))

        if any_missing and os.path.exists(csv_path):
            try:
                generate_all_plots(target_dir=plots_dir, verbose=False)
            except Exception as e:
                log_warning(f"Could not automatically generate plots: {e}")

        # Target dimensions for 2x2 grid layout inside dashboard
        target_w, target_h = 454, 250

        for key, path in required_plots.items():
            if os.path.exists(path):
                try:
                    img = pygame.image.load(path).convert_alpha()
                    scaled = pygame.transform.smoothscale(img, (target_w, target_h))
                    self.results_cached_plots[key] = scaled
                except Exception as e:
                    log_warning(f"Failed to load plot image {path}: {e}")
                    self.results_cached_plots[key] = None
            else:
                log_warning(f"Plot image missing from disk: {path}")
                self.results_cached_plots[key] = None

        self.results_plots_loaded = True

    def _open_results_folder(self) -> None:
        """Open the results directory in the user's OS file manager."""
        results_dir = os.path.abspath(config.RESULTS_DIR)
        os.makedirs(results_dir, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(results_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", results_dir])
            else:
                subprocess.Popen(["xdg-open", results_dir])
        except Exception as e:
            log_error(f"Error opening results folder '{results_dir}': {e}")

    def _draw_no_results_screen(self) -> None:
        """Render informative alert screen when no training results exist."""
        self.screen.fill(config.COLOR_BACKGROUND)

        # Header Title
        title = self.font_title.render("Training Results Dashboard", True, (15, 23, 42))
        self.screen.blit(title, (config.SCREEN_WIDTH // 2 - title.get_width() // 2, 70))

        # Alert Card Container
        card_w, card_h = 740, 390
        card_x = (config.SCREEN_WIDTH - card_w) // 2
        card_y = 150
        card_rect = pygame.Rect(card_x, card_y, card_w, card_h)

        pygame.draw.rect(self.screen, (255, 255, 255), card_rect, border_radius=14)
        pygame.draw.rect(self.screen, (254, 202, 202), card_rect, 2, border_radius=14)

        # Warning icon badge
        badge_y = card_y + 60
        pygame.draw.circle(self.screen, (254, 226, 226), (config.SCREEN_WIDTH // 2, badge_y), 32)
        w_surf = self.font_title.render("!", True, (220, 38, 38))
        self.screen.blit(w_surf, (config.SCREEN_WIDTH // 2 - w_surf.get_width() // 2, badge_y - w_surf.get_height() // 2 - 2))

        # Main alert text required by prompt
        alert_title = "No Training Results Found" if not self.results_error_msg or "not exist" in self.results_error_msg else "Training Results Unavailable"
        t1 = self.font_header.render(alert_title, True, (185, 28, 28))
        self.screen.blit(t1, (config.SCREEN_WIDTH // 2 - t1.get_width() // 2, card_y + 115))

        detail_text = self.results_error_msg or "The training dataset 'results/training_results.csv' does not exist yet."
        msg_lines = [
            detail_text,
            "Please train the agent first using Option [1] Train AI in the main menu,",
            "or run 'python main.py --train' directly from your terminal.",
            "Once training completes, this dashboard will visualize metrics and convergence curves.",
        ]

        curr_y = card_y + 165
        for line in msg_lines:
            l_surf = self.font_body.render(line, True, (71, 85, 105))
            self.screen.blit(l_surf, (config.SCREEN_WIDTH // 2 - l_surf.get_width() // 2, curr_y))
            curr_y += 24

        # Action Buttons: [Train AI Now] and [Back to Menu]
        mouse_pos = pygame.mouse.get_pos()
        self.btn_no_results_train.update(mouse_pos)
        self.btn_no_results_train.draw(self.screen)

        self.btn_no_results_back.update(mouse_pos)
        self.btn_no_results_back.draw(self.screen)

    def _draw_plot_card(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        title: str,
        surface: Optional[pygame.Surface],
        accent_color: Tuple[int, int, int],
    ) -> None:
        """Render a single Matplotlib plot card with header banner."""
        card_rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, (255, 255, 255), card_rect, border_radius=10)
        pygame.draw.rect(self.screen, (226, 232, 240), card_rect, 1, border_radius=10)

        # Header tag banner
        banner_h = 32
        banner_rect = pygame.Rect(x, y, w, banner_h)
        pygame.draw.rect(self.screen, (248, 250, 252), banner_rect, border_top_left_radius=10, border_top_right_radius=10)
        pygame.draw.line(self.screen, (226, 232, 240), (x, y + banner_h), (x + w, y + banner_h), 1)

        # Accent dot
        pygame.draw.circle(self.screen, accent_color, (x + 16, y + 16), 5)

        t_surf = self.font_card_title.render(title, True, (30, 41, 59))
        self.screen.blit(t_surf, (x + 28, y + 8))

        # Plot Surface
        if surface is not None:
            # Center image inside card below header
            img_x = x + (w - surface.get_width()) // 2
            img_y = y + banner_h + (h - banner_h - surface.get_height()) // 2
            self.screen.blit(surface, (img_x, img_y))
        else:
            fb_surf = self.font_body.render("Plot not available.", True, (148, 163, 184))
            self.screen.blit(fb_surf, (x + (w - fb_surf.get_width()) // 2, y + h // 2))

    def _draw_results_screen(self) -> None:
        """Render the complete Phase 12 Training Results Dashboard with scrolling and graphs."""
        self._load_results_data()
        self._load_results_plots()

        if self.results_metrics is None:
            self._draw_no_results_screen()
            return

        m = self.results_metrics
        mouse_pos = pygame.mouse.get_pos()

        # Fill background
        self.screen.fill(config.COLOR_BACKGROUND)

        # Viewport Setup & Clipping for Smooth Scrolling
        viewport_top = 80
        viewport_bottom = 675
        viewport_height = viewport_bottom - viewport_top  # 595 px
        viewport_rect = pygame.Rect(0, viewport_top, config.SCREEN_WIDTH, viewport_height)

        self.screen.set_clip(viewport_rect)

        scroll_y = int(self.results_scroll_pos)
        card_w = 960
        card_x = (config.SCREEN_WIDTH - card_w) // 2

        # 1. Seven Key Metric Cards (Row 1: 4 cards, Row 2: 3 cards)
        c1_w = (card_w - 3 * 16) // 4  # 228 px
        c1_h = 82
        r1_y = viewport_top + 16 - scroll_y

        row1_cards = [
            ("TRAINING EPISODES", f"{m['total_episodes']:,}", "Episodes completed", (37, 99, 235)),
            ("SUCCESS RATE", f"{m['sr_recent']:.1f}%", f"Overall: {m['sr_all']:.1f}%", config.COLOR_SUCCESS),
            ("AVERAGE REWARD", f"{m['rew_recent']:+.1f}", f"Overall: {m['rew_all']:+.1f}", config.COLOR_SUCCESS if m['rew_recent'] >= 0 else config.COLOR_DANGER),
            ("AVERAGE STEPS", f"{m['steps_recent']:.1f} steps", f"Optimal: 18 | All: {m['steps_all']:.1f}", (15, 23, 42)),
        ]

        for i, (title, val_str, sub_str, val_col) in enumerate(row1_cards):
            cx = card_x + i * (c1_w + 16)
            c_rect = pygame.Rect(cx, r1_y, c1_w, c1_h)
            pygame.draw.rect(self.screen, (255, 255, 255), c_rect, border_radius=10)
            pygame.draw.rect(self.screen, (226, 232, 240), c_rect, 1, border_radius=10)

            t_surf = self.font_card_title.render(title, True, (100, 116, 139))
            v_surf = self.font_val_large.render(val_str, True, val_col)
            s_surf = self.font_stat_sub.render(sub_str, True, (148, 163, 184))

            self.screen.blit(t_surf, (cx + 14, r1_y + 10))
            self.screen.blit(v_surf, (cx + 14, r1_y + 28))
            self.screen.blit(s_surf, (cx + 14, r1_y + 57))

        # Row 2 (3 cards)
        c2_w = (card_w - 2 * 18) // 3  # 308 px
        c2_h = 82
        r2_y = r1_y + c1_h + 12

        coll_col = config.COLOR_SUCCESS if m['coll_recent'] == 0 else (220, 38, 38)
        row2_cards = [
            ("COLLISION RATE", f"{m['coll_recent']:.2f} hits/ep", f"Overall: {m['coll_all']:.2f} hits/ep", coll_col),
            ("BEST EPISODE", f"Episode #{m['best_ep']}", f"Reward: {m['best_rew']:+.1f} ({m['best_steps']} steps)", (126, 34, 206)),
            ("FINAL EPSILON", f"{m['final_eps']:.4f}", "Exploration floor (5% residual)", (234, 88, 12)),
        ]

        for i, (title, val_str, sub_str, val_col) in enumerate(row2_cards):
            cx = card_x + i * (c2_w + 18)
            c_rect = pygame.Rect(cx, r2_y, c2_w, c2_h)
            pygame.draw.rect(self.screen, (255, 255, 255), c_rect, border_radius=10)
            pygame.draw.rect(self.screen, (226, 232, 240), c_rect, 1, border_radius=10)

            t_surf = self.font_card_title.render(title, True, (100, 116, 139))
            v_surf = self.font_val_large.render(val_str, True, val_col)
            s_surf = self.font_stat_sub.render(sub_str, True, (148, 163, 184))

            self.screen.blit(t_surf, (cx + 14, r2_y + 10))
            self.screen.blit(v_surf, (cx + 14, r2_y + 28))
            self.screen.blit(s_surf, (cx + 14, r2_y + 57))

        # 2. Section Header: Generated Graphs
        sec_y = r2_y + c2_h + 18
        pygame.draw.line(self.screen, (226, 232, 240), (card_x, sec_y), (card_x + card_w, sec_y), 1)

        sec_title = self.font_header.render("Generated Learning Curves & Evaluation Plots", True, (15, 23, 42))
        sec_sub = self.font_body.render(
            "Empirical training data from results/training_results.csv plotted using Matplotlib",
            True,
            (100, 116, 139),
        )
        self.screen.blit(sec_title, (card_x, sec_y + 10))
        self.screen.blit(sec_sub, (card_x, sec_y + 34))

        # 3. Four Generated Graphs (2x2 Grid)
        plot_box_w = (card_w - 20) // 2  # 470 px
        plot_box_h = 295
        grid_gap = 20

        # Row 1 plots
        p_row1_y = sec_y + 60
        self._draw_plot_card(
            card_x,
            p_row1_y,
            plot_box_w,
            plot_box_h,
            "1. Reward vs Episode (Convergence to +170.0)",
            self.results_cached_plots.get("reward_vs_episode"),
            (37, 99, 235),
        )
        self._draw_plot_card(
            card_x + plot_box_w + grid_gap,
            p_row1_y,
            plot_box_w,
            plot_box_h,
            "2. Steps vs Episode (Convergence to 18 Steps)",
            self.results_cached_plots.get("steps_vs_episode"),
            (16, 185, 129),
        )

        # Row 2 plots
        p_row2_y = p_row1_y + plot_box_h + 16
        self._draw_plot_card(
            card_x,
            p_row2_y,
            plot_box_w,
            plot_box_h,
            "3. Success Rate vs Episode (Rolling Window to 100%)",
            self.results_cached_plots.get("success_rate"),
            (147, 51, 234),
        )
        self._draw_plot_card(
            card_x + plot_box_w + grid_gap,
            p_row2_y,
            plot_box_w,
            plot_box_h,
            "4. Collision Rate vs Episode (Avoidance to 0 Hits)",
            self.results_cached_plots.get("collision_rate"),
            (234, 88, 12),
        )

        # 4. Methodology / Observation Callout Box
        callout_y = p_row2_y + plot_box_h + 16
        callout_h = 66
        callout_rect = pygame.Rect(card_x, callout_y, card_w, callout_h)
        pygame.draw.rect(self.screen, (241, 245, 249), callout_rect, border_radius=8)
        pygame.draw.rect(self.screen, (203, 213, 225), callout_rect, 1, border_radius=8)

        ct_surf = self.font_card_title.render("Q-LEARNING CONVERGENCE OBSERVATION:", True, (51, 65, 85))
        c1_surf = self.font_stat_sub.render(
            "• Early exploration (epsilon = 1.0 -> 0.4) incurs frequent collisions (-10) and boundary violations while discovering the campus layout.",
            True,
            (71, 85, 105),
        )
        c2_surf = self.font_stat_sub.render(
            "• Late training (epsilon = 0.05) demonstrates stable Bellman optimality: 100% success rate, 0 collisions, and the exact 18-step shortest path.",
            True,
            (71, 85, 105),
        )
        self.screen.blit(ct_surf, (card_x + 16, callout_y + 8))
        self.screen.blit(c1_surf, (card_x + 16, callout_y + 26))
        self.screen.blit(c2_surf, (card_x + 16, callout_y + 44))

        # Total content height calculation
        total_content_bottom = callout_y + callout_h + 20 + scroll_y
        self.results_max_scroll = max(0.0, total_content_bottom - viewport_bottom)

        # Restore Screen Clip for Fixed Overlays
        self.screen.set_clip(None)

        # 5. Fixed Top Header Bar (Always Visible)
        header_rect = pygame.Rect(0, 0, config.SCREEN_WIDTH, viewport_top)
        pygame.draw.rect(self.screen, config.COLOR_BACKGROUND, header_rect)
        pygame.draw.line(self.screen, (203, 213, 225), (0, viewport_top), (config.SCREEN_WIDTH, viewport_top), 1)

        title_surf = self.font_title.render("Training Results Dashboard", True, (15, 23, 42))
        sub_surf = self.font_subtitle.render("Empirical Q-Learning Training Telemetry & Generated Learning Curves", True, (71, 85, 105))
        self.screen.blit(title_surf, (card_x, 16))
        self.screen.blit(sub_surf, (card_x, 50))

        # Scroll instructions badge
        if self.results_max_scroll > 0:
            badge_w, badge_h = 220, 26
            badge_x = card_x + card_w - badge_w
            badge_y = 26
            pygame.draw.rect(self.screen, (241, 245, 249), (badge_x, badge_y, badge_w, badge_h), border_radius=13)
            pygame.draw.rect(self.screen, (203, 213, 225), (badge_x, badge_y, badge_w, badge_h), 1, border_radius=13)
            sc_surf = self.font_stat_sub.render("Scroll: Wheel / Up / Down / PgDn", True, (71, 85, 105))
            self.screen.blit(sc_surf, (badge_x + (badge_w - sc_surf.get_width()) // 2, badge_y + 5))

        # 6. Fixed Bottom Action Bar (Always Visible)
        bar_rect = pygame.Rect(0, viewport_bottom, config.SCREEN_WIDTH, config.SCREEN_HEIGHT - viewport_bottom)
        pygame.draw.rect(self.screen, (255, 255, 255), bar_rect)
        pygame.draw.line(self.screen, (226, 232, 240), (0, viewport_bottom), (config.SCREEN_WIDTH, viewport_bottom), 1)

        self.btn_open_folder.update(mouse_pos)
        self.btn_open_folder.draw(self.screen)

        self.btn_results_back.update(mouse_pos)
        self.btn_results_back.draw(self.screen)

        # 7. Vertical Scrollbar
        if self.results_max_scroll > 0:
            track_x = 1068
            track_y = viewport_top + 10
            track_w = 8
            track_h = viewport_height - 20

            pygame.draw.rect(self.screen, (241, 245, 249), (track_x, track_y, track_w, track_h), border_radius=4)

            total_h = viewport_height + self.results_max_scroll
            thumb_h = max(36, int(track_h * (viewport_height / total_h)))
            scroll_ratio = self.results_scroll_pos / self.results_max_scroll
            thumb_y = track_y + int((track_h - thumb_h) * scroll_ratio)

            thumb_color = (100, 116, 139) if self.is_dragging_scrollbar else (148, 163, 184)
            pygame.draw.rect(self.screen, thumb_color, (track_x, thumb_y, track_w, thumb_h), border_radius=4)

    # --------------------------------------------------------------------------
    # Sub-Screen: [5] How RL Works (Viva Theory Guide)
    # --------------------------------------------------------------------------

    def _draw_arrow(
        self,
        start_pos: Tuple[int, int],
        end_pos: Tuple[int, int],
        color: Tuple[int, int, int] = (148, 163, 184),
        width: int = 2,
    ) -> None:
        """Procedurally render a crisp directional arrow between two coordinates."""
        pygame.draw.line(self.screen, color, start_pos, end_pos, width)
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        length = (dx ** 2 + dy ** 2) ** 0.5
        if length > 0:
            ux = dx / length
            uy = dy / length
            px = -uy
            py = ux
            head_size = 6
            p1 = (end_pos[0] - head_size * ux + head_size * 0.6 * px, end_pos[1] - head_size * uy + head_size * 0.6 * py)
            p2 = (end_pos[0] - head_size * ux - head_size * 0.6 * px, end_pos[1] - head_size * uy - head_size * 0.6 * py)
            pygame.draw.polygon(self.screen, color, [end_pos, p1, p2])

    def _draw_rl_guide_screen(self) -> None:
        """
        Render Phase 13: "How RL Works" comprehensive educational screen.
        Displays the RL Loop, the Q-Learning Bellman Equation, and the 10 Core Concepts.
        """
        self.screen.fill(config.COLOR_BACKGROUND)
        mouse_pos = pygame.mouse.get_pos()

        # Viewport Setup & Clipping for Smooth Scrolling
        viewport_top = 80
        viewport_bottom = 675
        viewport_height = viewport_bottom - viewport_top  # 595 px
        viewport_rect = pygame.Rect(0, viewport_top, config.SCREEN_WIDTH, viewport_height)

        self.screen.set_clip(viewport_rect)

        scroll_y = int(self.rl_scroll_pos)
        card_w = 960
        card_x = (config.SCREEN_WIDTH - card_w) // 2  # 70 px

        # ----------------------------------------------------------------------
        # 1. The Reinforcement Learning Interaction Loop (8 Stages)
        # ----------------------------------------------------------------------
        sec1_y = viewport_top + 14 - scroll_y
        sec1_title = self.font_header.render("1. The Reinforcement Learning Interaction Loop", True, (15, 23, 42))
        sec1_sub = self.font_body.render(
            "Continuous cyclic interaction between Agent and Campus Environment executing on every step",
            True,
            (100, 116, 139),
        )
        self.screen.blit(sec1_title, (card_x, sec1_y))
        self.screen.blit(sec1_sub, (card_x, sec1_y + 24))

        loop_box_y = sec1_y + 48
        loop_box_h = 162
        loop_rect = pygame.Rect(card_x, loop_box_y, card_w, loop_box_h)
        pygame.draw.rect(self.screen, (255, 255, 255), loop_rect, border_radius=12)
        pygame.draw.rect(self.screen, (226, 232, 240), loop_rect, 1, border_radius=12)

        # 8 Cycle Stages layout: Top Row (1 -> 4) and Bottom Row (8 <- 5)
        # Top Row: [1. Environment] -> [2. State] -> [3. Agent] -> [4. Action]
        # Bottom Row: [8. New State] <- [7. Q-Table Update] <- [6. Reward] <- [5. Environment]
        stage_w = 196
        stage_h = 56
        gap_w = 45

        top_stages = [
            ("1. Environment", "Campus Grid (10x10)", (16, 185, 129)),
            ("2. State", "Current Pos (row, col)", (99, 102, 241)),
            ("3. Agent", "Q-Learning Student", (37, 99, 235)),
            ("4. Action", "Move UP / DN / LT / RT", (225, 29, 72)),
        ]
        r1_y = loop_box_y + 16

        for i, (title, sub_text, col) in enumerate(top_stages):
            sx = card_x + 18 + i * (stage_w + gap_w)
            s_rect = pygame.Rect(sx, r1_y, stage_w, stage_h)
            pygame.draw.rect(self.screen, (248, 250, 252), s_rect, border_radius=8)
            pygame.draw.rect(self.screen, (226, 232, 240), s_rect, 1, border_radius=8)
            pygame.draw.rect(self.screen, col, (sx, r1_y, 4, stage_h), border_top_left_radius=8, border_bottom_left_radius=8)

            t_surf = self.font_card_title.render(title, True, (15, 23, 42))
            st_surf = self.font_stat_sub.render(sub_text, True, (100, 116, 139))
            self.screen.blit(t_surf, (sx + 12, r1_y + 10))
            self.screen.blit(st_surf, (sx + 12, r1_y + 30))

            # Procedural Right Arrow between top stages
            if i < 3:
                start_p = (sx + stage_w + 6, r1_y + stage_h // 2)
                end_p = (sx + stage_w + gap_w - 6, r1_y + stage_h // 2)
                self._draw_arrow(start_p, end_p, (148, 163, 184))

        # Procedural Down Arrow from Action (4) to Environment (5)
        d_x = card_x + 18 + 3 * (stage_w + gap_w) + stage_w // 2
        d_start = (d_x, r1_y + stage_h + 3)
        d_end = (d_x, r1_y + stage_h + 17)
        self._draw_arrow(d_start, d_end, (148, 163, 184))

        # Bottom Row: [8. New State] <- [7. Q-Table Update] <- [6. Reward] <- [5. Environment]
        bottom_stages = [
            ("8. New State", "Transition to (r', c')", (99, 102, 241)),
            ("7. Q-Table Update", "Bellman TD Equation", (147, 51, 234)),
            ("6. Reward", "+100 Goal / +5 / -10", (217, 119, 6)),
            ("5. Environment", "Calculates Dynamics", (16, 185, 129)),
        ]
        r2_y = r1_y + stage_h + 20

        for i, (title, sub_text, col) in enumerate(bottom_stages):
            sx = card_x + 18 + i * (stage_w + gap_w)
            s_rect = pygame.Rect(sx, r2_y, stage_w, stage_h)
            pygame.draw.rect(self.screen, (248, 250, 252), s_rect, border_radius=8)
            pygame.draw.rect(self.screen, (226, 232, 240), s_rect, 1, border_radius=8)
            pygame.draw.rect(self.screen, col, (sx, r2_y, 4, stage_h), border_top_left_radius=8, border_bottom_left_radius=8)

            t_surf = self.font_card_title.render(title, True, (15, 23, 42))
            st_surf = self.font_stat_sub.render(sub_text, True, (100, 116, 139))
            self.screen.blit(t_surf, (sx + 12, r2_y + 10))
            self.screen.blit(st_surf, (sx + 12, r2_y + 30))

            # Procedural Left Arrow between bottom stages (flowing from right to left)
            if i < 3:
                start_p = (sx + stage_w + gap_w - 6, r2_y + stage_h // 2)
                end_p = (sx + stage_w + 6, r2_y + stage_h // 2)
                self._draw_arrow(start_p, end_p, (148, 163, 184))

        # Procedural Upward Loop Connector from New State (8) back to Environment (1)
        up_x = card_x + 18 + stage_w // 2
        up_start = (up_x, r2_y - 3)
        up_end = (up_x, r1_y + stage_h + 3)
        self._draw_arrow(up_start, up_end, (148, 163, 184))

        # ----------------------------------------------------------------------
        # 2. Q-Learning Bellman Optimality Equation
        # ----------------------------------------------------------------------
        sec2_y = loop_box_y + loop_box_h + 18
        sec2_title = self.font_header.render("2. The Q-Learning Equation (Bellman Optimality)", True, (15, 23, 42))
        sec2_sub = self.font_body.render(
            "Temporal Difference (TD) control rule updating the expected return matrix Q(s, a)",
            True,
            (100, 116, 139),
        )
        self.screen.blit(sec2_title, (card_x, sec2_y))
        self.screen.blit(sec2_sub, (card_x, sec2_y + 24))

        eq_box_y = sec2_y + 48
        eq_box_h = 168
        eq_rect = pygame.Rect(card_x, eq_box_y, card_w, eq_box_h)
        pygame.draw.rect(self.screen, (255, 255, 255), eq_rect, border_radius=12)
        pygame.draw.rect(self.screen, (226, 232, 240), eq_rect, 1, border_radius=12)

        # Formula Banner
        fb_h = 44
        fb_rect = pygame.Rect(card_x + 16, eq_box_y + 14, card_w - 32, fb_h)
        pygame.draw.rect(self.screen, (15, 23, 42), fb_rect, border_radius=8)

        formula_text = "Q(s, a) = Q(s, a) + α [ r + γ · max_a' Q(s', a') - Q(s, a) ]"
        f_surf = self.font_formula.render(formula_text, True, (241, 245, 249))
        self.screen.blit(f_surf, (fb_rect.centerx - f_surf.get_width() // 2, fb_rect.centery - f_surf.get_height() // 2))

        # Explanation 3-Column Parameter Cards
        param_cards = [
            (
                "α = Learning Rate",
                "Default: 0.1",
                "Controls step size: determines how heavily new experience overrides prior Q-value estimates (0 < α ≤ 1).",
                (37, 99, 235),
            ),
            (
                "γ = Discount Factor",
                "Default: 0.9",
                "Quantifies importance of future delayed rewards vs immediate reward (0 = myopic, 0.9 = strategic foresight).",
                (147, 51, 234),
            ),
            (
                "r = Reward",
                "+100 / +5 / -10",
                "Immediate numerical feedback signal received after transitioning from state s to state s' via action a.",
                (217, 119, 6),
            ),
        ]

        p_w = (card_w - 32 - 2 * 14) // 3  # 294 px
        p_h = 76
        py = eq_box_y + 14 + fb_h + 12

        for i, (p_title, p_badge, p_desc, p_col) in enumerate(param_cards):
            px = card_x + 16 + i * (p_w + 14)
            p_rect = pygame.Rect(px, py, p_w, p_h)
            pygame.draw.rect(self.screen, (248, 250, 252), p_rect, border_radius=8)
            pygame.draw.rect(self.screen, (226, 232, 240), p_rect, 1, border_radius=8)
            pygame.draw.rect(self.screen, p_col, (px, py, 4, p_h), border_top_left_radius=8, border_bottom_left_radius=8)

            pt_surf = self.font_body_bold.render(p_title, True, p_col)
            pb_surf = self.font_concept_num.render(p_badge, True, (100, 116, 139))
            self.screen.blit(pt_surf, (px + 12, py + 8))
            self.screen.blit(pb_surf, (px + p_w - pb_surf.get_width() - 12, py + 10))

            words = p_desc.split(" ")
            line1, line2 = "", ""
            for word in words:
                if len(line1 + " " + word) < 44:
                    line1 = (line1 + " " + word).strip()
                else:
                    line2 = (line2 + " " + word).strip()

            l1_s = self.font_stat_sub.render(line1, True, (71, 85, 105))
            l2_s = self.font_stat_sub.render(line2, True, (71, 85, 105))
            self.screen.blit(l1_s, (px + 12, py + 30))
            self.screen.blit(l2_s, (px + 12, py + 48))

        # ----------------------------------------------------------------------
        # 3. The 10 Fundamental Reinforcement Learning Concepts
        # ----------------------------------------------------------------------
        sec3_y = eq_box_y + eq_box_h + 18
        sec3_title = self.font_header.render("3. The 10 Fundamental Reinforcement Learning Concepts", True, (15, 23, 42))
        sec3_sub = self.font_body.render(
            "Clear, concise definitions and examination viva answers mapped to this campus project",
            True,
            (100, 116, 139),
        )
        self.screen.blit(sec3_title, (card_x, sec3_y))
        self.screen.blit(sec3_sub, (card_x, sec3_y + 24))

        concepts = [
            (
                "1. Agent (Student)",
                "The active learner and decision-maker navigating campus. It observes coordinates, takes moves, and learns through trial-and-error.",
                (37, 99, 235),
            ),
            (
                "2. Environment (Campus)",
                "The 10x10 spatial gridworld the agent interacts with, containing 18 impassable buildings, safe cafeterias, and mud hazard zones.",
                (16, 185, 129),
            ),
            (
                "3. State (Coordinate)",
                "The current spatial position (row, col) of the student, mapped to 100 unique discrete state indices: s = row * 10 + col (0 to 99).",
                (99, 102, 241),
            ),
            (
                "4. Action (Movement)",
                "The 4 cardinal directional decisions available at each step: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT. Valid actions shift to adjacent cells.",
                (225, 29, 72),
            ),
            (
                "5. Reward (Feedback)",
                "Numerical feedback guiding optimization: +100 for Library, +5 moving closer, -3 moving farther, -10 for wall collisions or hazards.",
                (217, 119, 6),
            ),
            (
                "6. Q-Table (Memory)",
                "A 2D NumPy lookup table of shape (100, 4) initialized to zeros, storing expected cumulative return Q(s, a) for every possible move.",
                (147, 51, 234),
            ),
            (
                "7. Exploration (Discovery)",
                "Choosing random actions with probability ε (starts at 1.0). Enables the agent to discover alternative paths and avoid local optima.",
                (14, 165, 233),
            ),
            (
                "8. Exploitation (Greedy)",
                "Choosing the best known action with probability 1 - ε: argmax_a Q(s, a). Uses acquired knowledge to navigate efficiently to the goal.",
                (13, 148, 136),
            ),
            (
                "9. Episode (Single Trial)",
                "One complete attempt starting from Main Gate (0, 0) until reaching Central Library (9, 9) or reaching 200 steps limit. Training = 1,000 eps.",
                (234, 88, 12),
            ),
            (
                "10. Learning (Convergence)",
                "Iterative updating of the Q-table using the Bellman equation, decaying ε (0.995 factor) until converging to the optimal 18-step route.",
                (5, 150, 105),
            ),
        ]

        c_w = (card_w - 18) // 2  # 471 px
        c_h = 86
        c_start_y = sec3_y + 48

        for idx, (c_title, c_desc, c_col) in enumerate(concepts):
            col_idx = idx % 2
            row_idx = idx // 2
            cx = card_x + col_idx * (c_w + 18)
            cy = c_start_y + row_idx * (c_h + 12)

            c_rect = pygame.Rect(cx, cy, c_w, c_h)
            pygame.draw.rect(self.screen, (255, 255, 255), c_rect, border_radius=10)
            pygame.draw.rect(self.screen, (226, 232, 240), c_rect, 1, border_radius=10)
            pygame.draw.rect(self.screen, c_col, (cx, cy, 5, c_h), border_top_left_radius=10, border_bottom_left_radius=10)

            t_surf = self.font_body_bold.render(c_title, True, (15, 23, 42))
            self.screen.blit(t_surf, (cx + 16, cy + 10))

            words = c_desc.split(" ")
            lines = []
            curr_line = ""
            for word in words:
                if len(curr_line + " " + word) < 56:
                    curr_line = (curr_line + " " + word).strip()
                else:
                    lines.append(curr_line)
                    curr_line = word
            if curr_line:
                lines.append(curr_line)

            for line_i, line_text in enumerate(lines[:3]):
                l_surf = self.font_stat_sub.render(line_text, True, (71, 85, 105))
                self.screen.blit(l_surf, (cx + 16, cy + 32 + line_i * 16))

        # ----------------------------------------------------------------------
        # 4. Viva Examination Summary Callout
        # ----------------------------------------------------------------------
        last_row_bottom = c_start_y + 5 * (c_h + 12)
        viva_box_y = last_row_bottom + 6
        viva_box_h = 66
        viva_rect = pygame.Rect(card_x, viva_box_y, card_w, viva_box_h)
        pygame.draw.rect(self.screen, (240, 253, 250), viva_rect, border_radius=8)
        pygame.draw.rect(self.screen, (204, 251, 241), viva_rect, 1, border_radius=8)

        vt_surf = self.font_card_title.render("COLLEGE VIVA EXAM KEY TAKEAWAY:", True, (15, 118, 110))
        vl1_surf = self.font_stat_sub.render(
            "• Q-Learning is 'model-free' because it doesn't need transition probabilities P(s'|s,a), and 'off-policy' because it evaluates greedy max future return.",
            True,
            (19, 78, 74),
        )
        vl2_surf = self.font_stat_sub.render(
            "• Terminal State Rule: At goal (9, 9), max Q(s', a') = 0 because no future transitions occur, avoiding phantom reward inflation.",
            True,
            (19, 78, 74),
        )
        self.screen.blit(vt_surf, (card_x + 16, viva_box_y + 8))
        self.screen.blit(vl1_surf, (card_x + 16, viva_box_y + 26))
        self.screen.blit(vl2_surf, (card_x + 16, viva_box_y + 44))

        # Dynamic max scroll calculation
        total_content_bottom = viva_box_y + viva_box_h + 24 + scroll_y
        self.rl_max_scroll = max(0.0, total_content_bottom - viewport_bottom)

        # Restore Screen Clip
        self.screen.set_clip(None)

        # ----------------------------------------------------------------------
        # 5. Fixed Top Header Bar
        # ----------------------------------------------------------------------
        header_rect = pygame.Rect(0, 0, config.SCREEN_WIDTH, viewport_top)
        pygame.draw.rect(self.screen, config.COLOR_BACKGROUND, header_rect)
        pygame.draw.line(self.screen, (203, 213, 225), (0, viewport_top), (config.SCREEN_WIDTH, viewport_top), 1)

        title_surf = self.font_title.render("How Reinforcement Learning Works", True, (15, 23, 42))
        sub_surf = self.font_subtitle.render("Theoretical Foundations, RL Cycle, Bellman Optimality Equation & 10 Core Concepts", True, (71, 85, 105))
        self.screen.blit(title_surf, (card_x, 16))
        self.screen.blit(sub_surf, (card_x, 50))

        if self.rl_max_scroll > 0:
            badge_w, badge_h = 220, 26
            badge_x = card_x + card_w - badge_w
            badge_y = 26
            pygame.draw.rect(self.screen, (241, 245, 249), (badge_x, badge_y, badge_w, badge_h), border_radius=13)
            pygame.draw.rect(self.screen, (203, 213, 225), (badge_x, badge_y, badge_w, badge_h), 1, border_radius=13)
            sc_surf = self.font_stat_sub.render("Scroll: Wheel / Up / Down / PgDn", True, (71, 85, 105))
            self.screen.blit(sc_surf, (badge_x + (badge_w - sc_surf.get_width()) // 2, badge_y + 5))

        # ----------------------------------------------------------------------
        # 6. Fixed Bottom Action Bar
        # ----------------------------------------------------------------------
        bar_rect = pygame.Rect(0, viewport_bottom, config.SCREEN_WIDTH, config.SCREEN_HEIGHT - viewport_bottom)
        pygame.draw.rect(self.screen, (255, 255, 255), bar_rect)
        pygame.draw.line(self.screen, (226, 232, 240), (0, viewport_bottom), (config.SCREEN_WIDTH, viewport_bottom), 1)

        self.btn_rl_back.update(mouse_pos)
        self.btn_rl_back.draw(self.screen)

        # ----------------------------------------------------------------------
        # 7. Vertical Scrollbar
        # ----------------------------------------------------------------------
        if self.rl_max_scroll > 0:
            track_x = 1068
            track_y = viewport_top + 10
            track_w = 8
            track_h = viewport_height - 20

            pygame.draw.rect(self.screen, (241, 245, 249), (track_x, track_y, track_w, track_h), border_radius=4)

            total_h = viewport_height + self.rl_max_scroll
            thumb_h = max(36, int(track_h * (viewport_height / total_h)))
            scroll_ratio = self.rl_scroll_pos / self.rl_max_scroll
            thumb_y = track_y + int((track_h - thumb_h) * scroll_ratio)

            thumb_color = (100, 116, 139) if self.is_dragging_rl_scrollbar else (148, 163, 184)
            pygame.draw.rect(self.screen, thumb_color, (track_x, thumb_y, track_w, thumb_h), border_radius=4)

    # --------------------------------------------------------------------------
    # Sub-Screen: [6] Settings Modal (Phase 14)
    # --------------------------------------------------------------------------

    def _load_settings_into_form(self) -> None:
        """Load active or saved configuration into settings form state."""
        self.settings_data = config.load_config()
        self.settings_active_field = None
        self.settings_text_buffer = ""
        self.settings_status = ("normal", "Adjust hyperparameters and simulation parameters below.")

    def _commit_active_text_field(self) -> None:
        """Parse and commit text buffer of active input field."""
        if not self.settings_active_field:
            return
        field = self.settings_active_field
        buf = self.settings_text_buffer.strip()
        try:
            if field in ("grid_size", "training_episodes", "max_steps_per_episode"):
                val = int(buf)
                self.settings_data[field] = val
            elif field in ("learning_rate", "discount_factor", "initial_epsilon", "min_epsilon", "epsilon_decay"):
                val = float(buf)
                self.settings_data[field] = val
            self.settings_status = ("normal", f"Updated {field} to {buf}")
        except ValueError:
            self.settings_status = ("error", f"Invalid numeric input '{buf}' for {field}.")
        self.settings_active_field = None
        self.settings_text_buffer = ""

    def _save_settings_from_form(self) -> None:
        """Validate and persist current settings form to config.json."""
        self._commit_active_text_field()
        valid, msg = config.validate_config(self.settings_data)
        if not valid:
            self.settings_status = ("error", f"Validation Error: {msg}")
            return

        ok, save_msg = config.save_config(self.settings_data)
        if ok:
            self.settings_status = ("success", "Settings saved successfully to config.json and applied to runtime!")
            if self.on_settings_saved:
                try:
                    self.on_settings_saved()
                except Exception as e:
                    print(f"[Settings] Callback error: {e}")
        else:
            self.settings_status = ("error", f"Save Error: {save_msg}")

    def _reset_settings_to_defaults(self) -> None:
        """Reset form and runtime config to factory defaults."""
        self.settings_active_field = None
        self.settings_text_buffer = ""
        self.settings_data = config.reset_to_defaults()
        self.settings_status = ("info", "Restored factory defaults and saved to config.json.")
        if self.on_settings_saved:
            try:
                self.on_settings_saved()
            except Exception as e:
                print(f"[Settings] Callback error: {e}")

    def _draw_settings_screen(self) -> None:
        """Render Phase 14: Settings & Hyperparameter Configuration Screen."""
        self.screen.fill(config.COLOR_BACKGROUND)
        mouse_pos = pygame.mouse.get_pos()

        # 1. Header Banner
        title = self.font_title.render("Application Settings", True, (15, 23, 42))
        sub = self.font_subtitle.render(
            "Configure Tabular Q-Learning Hyperparameters, Campus Environment, and Simulation Rates",
            True,
            (71, 85, 105),
        )
        self.screen.blit(title, (70, 16))
        self.screen.blit(sub, (70, 52))

        # Horizontal divider under header
        pygame.draw.line(self.screen, (203, 213, 225), (0, 78), (config.SCREEN_WIDTH, 78), 1)

        # 2. Two-Column Layout Setup
        col1_x = 70
        col1_w = 465
        col2_x = 565
        col2_w = 465

        # Column 1 Header
        c1_head = self.font_card_title.render("ENVIRONMENT & SIMULATION SETTINGS", True, (37, 99, 235))
        self.screen.blit(c1_head, (col1_x, 88))

        # Column 2 Header
        c2_head = self.font_card_title.render("Q-LEARNING ALGORITHM HYPERPARAMETERS", True, (147, 51, 234))
        self.screen.blit(c2_head, (col2_x, 88))

        # ----------------------------------------------------------------------
        # Field Definitions for Left Column (Environment & Training)
        # ----------------------------------------------------------------------
        left_fields = [
            ("grid_size", "Campus Grid Size", "Safely Supported: 8x8 to 16x16 (10x10 standard)", True, 2, 6, 16, 0),
            ("max_steps_per_episode", "Max Steps / Episode", "Truncation limit before episode failure (default: 200)", True, 25, 25, 1000, 0),
            ("training_episodes", "Training Episodes", "Total episodes for agent learning (default: 1000)", True, 100, 50, 20000, 0),
        ]

        row_h = 68
        gap_y = 10
        start_y = 108

        # Store rects for click interaction
        self.settings_rects = {}

        for idx, (key, label, subtitle, is_int, step, min_v, max_v, dec) in enumerate(left_fields):
            cy = start_y + idx * (row_h + gap_y)
            c_rect = pygame.Rect(col1_x, cy, col1_w, row_h)
            pygame.draw.rect(self.screen, (255, 255, 255), c_rect, border_radius=8)
            pygame.draw.rect(self.screen, (226, 232, 240), c_rect, 1, border_radius=8)
            pygame.draw.rect(self.screen, (37, 99, 235), (col1_x, cy, 4, row_h), border_top_left_radius=8, border_bottom_left_radius=8)

            # Text Labels
            l_surf = self.font_body_bold.render(label, True, (15, 23, 42))
            s_surf = self.font_stat_sub.render(subtitle, True, (100, 116, 139))
            self.screen.blit(l_surf, (col1_x + 14, cy + 12))
            self.screen.blit(s_surf, (col1_x + 14, cy + 34))

            # Stepper & Value Controls
            btn_w = 30
            btn_h = 32
            val_w = 90
            ctrl_y = cy + (row_h - btn_h) // 2
            minus_x = col1_x + col1_w - 14 - (btn_w * 2 + val_w + 8)
            val_x = minus_x + btn_w + 4
            plus_x = val_x + val_w + 4

            minus_rect = pygame.Rect(minus_x, ctrl_y, btn_w, btn_h)
            val_rect = pygame.Rect(val_x, ctrl_y, val_w, btn_h)
            plus_rect = pygame.Rect(plus_x, ctrl_y, btn_w, btn_h)

            self.settings_rects[f"{key}_minus"] = (minus_rect, key, "minus", step, min_v, max_v, dec)
            self.settings_rects[f"{key}_plus"] = (plus_rect, key, "plus", step, min_v, max_v, dec)
            self.settings_rects[f"{key}_val"] = (val_rect, key, "val", dec)

            # Draw Minus Button
            m_hover = minus_rect.collidepoint(mouse_pos)
            pygame.draw.rect(self.screen, (226, 232, 240) if m_hover else (241, 245, 249), minus_rect, border_radius=6)
            pygame.draw.rect(self.screen, (203, 213, 225), minus_rect, 1, border_radius=6)
            m_text = self.font_btn.render("-", True, (30, 41, 59))
            self.screen.blit(m_text, (minus_rect.centerx - m_text.get_width() // 2, minus_rect.centery - m_text.get_height() // 2 - 1))

            # Draw Value Box
            is_active = (self.settings_active_field == key)
            pygame.draw.rect(self.screen, (255, 255, 255), val_rect, border_radius=6)
            if is_active:
                pygame.draw.rect(self.screen, (37, 99, 235), val_rect, 2, border_radius=6)
            else:
                pygame.draw.rect(self.screen, (203, 213, 225), val_rect, 1, border_radius=6)

            cur_val = self.settings_data.get(key, min_v)
            if is_active:
                display_str = self.settings_text_buffer + ("|" if (pygame.time.get_ticks() // 500) % 2 == 0 else "")
            else:
                if key == "grid_size":
                    display_str = f"{cur_val} x {cur_val}"
                elif is_int:
                    display_str = str(cur_val)
                else:
                    display_str = f"{cur_val:.{dec}f}"

            v_surf = self.font_val.render(display_str, True, (15, 23, 42) if not is_active else (37, 99, 235))
            self.screen.blit(v_surf, (val_rect.centerx - v_surf.get_width() // 2, val_rect.centery - v_surf.get_height() // 2))

            # Draw Plus Button
            p_hover = plus_rect.collidepoint(mouse_pos)
            pygame.draw.rect(self.screen, (226, 232, 240) if p_hover else (241, 245, 249), plus_rect, border_radius=6)
            pygame.draw.rect(self.screen, (203, 213, 225), plus_rect, 1, border_radius=6)
            p_text = self.font_btn.render("+", True, (30, 41, 59))
            self.screen.blit(p_text, (plus_rect.centerx - p_text.get_width() // 2, plus_rect.centery - p_text.get_height() // 2 - 1))

        # Row 4: Training Speed (Segmented Buttons)
        cy = start_y + 3 * (row_h + gap_y)
        c_rect = pygame.Rect(col1_x, cy, col1_w, row_h)
        pygame.draw.rect(self.screen, (255, 255, 255), c_rect, border_radius=8)
        pygame.draw.rect(self.screen, (226, 232, 240), c_rect, 1, border_radius=8)
        pygame.draw.rect(self.screen, (37, 99, 235), (col1_x, cy, 4, row_h), border_top_left_radius=8, border_bottom_left_radius=8)

        l_surf = self.font_body_bold.render("Training Speed", True, (15, 23, 42))
        s_surf = self.font_stat_sub.render("Default simulation rate for live animated training mode", True, (100, 116, 139))
        self.screen.blit(l_surf, (col1_x + 14, cy + 12))
        self.screen.blit(s_surf, (col1_x + 14, cy + 34))

        # 3 Speed Segmented Buttons
        speeds = ["SLOW", "NORMAL", "FAST"]
        cur_speed = str(self.settings_data.get("training_speed", "NORMAL")).upper()
        total_sp_w = 175
        sp_x = col1_x + col1_w - 14 - total_sp_w
        sp_y = cy + (row_h - 32) // 2
        btn_w_each = total_sp_w // 3

        for i, sp in enumerate(speeds):
            sp_rect = pygame.Rect(sp_x + i * btn_w_each, sp_y, btn_w_each, 32)
            self.settings_rects[f"speed_{sp}"] = (sp_rect, sp)
            is_cur = (cur_speed == sp)
            sp_hover = sp_rect.collidepoint(mouse_pos)

            if is_cur:
                bg_col = (37, 99, 235)
                tx_col = (255, 255, 255)
            elif sp_hover:
                bg_col = (226, 232, 240)
                tx_col = (15, 23, 42)
            else:
                bg_col = (241, 245, 249)
                tx_col = (71, 85, 105)

            pygame.draw.rect(self.screen, bg_col, sp_rect)
            pygame.draw.rect(self.screen, (203, 213, 225), sp_rect, 1)
            sp_surf = self.font_stat_sub.render(sp, True, tx_col)
            self.screen.blit(sp_surf, (sp_rect.centerx - sp_surf.get_width() // 2, sp_rect.centery - sp_surf.get_height() // 2))

        # Row 5: Safe Zones, Crowded Zones & Hazards Summary Card (Phase 15)
        cy = start_y + 4 * (row_h + gap_y)
        c_rect = pygame.Rect(col1_x, cy, col1_w, row_h)
        pygame.draw.rect(self.screen, (240, 253, 250), c_rect, border_radius=8)
        pygame.draw.rect(self.screen, (204, 251, 241), c_rect, 1, border_radius=8)
        pygame.draw.rect(self.screen, (15, 118, 110), (col1_x, cy, 4, row_h), border_top_left_radius=8, border_bottom_left_radius=8)

        z_title = self.font_card_title.render("CAMPUS ENVIRONMENT DYNAMICS (PHASE 15 ENHANCED)", True, (15, 118, 110))
        z_desc1 = self.font_stat_sub.render("• Gate (0,0)  |  Central Library (+100)  |  18 Impassable Buildings (-10)", True, (19, 78, 74))
        z_desc2 = self.font_stat_sub.render(f"• Safe Zones (+{config.REWARD_SAFE_ZONE:.0f})  |  Crowded Zones ({config.REWARD_CROWDED_ZONE:.0f})  |  Hazard Zones ({config.REWARD_UNSAFE_ZONE:.0f})", True, (19, 78, 74))
        self.screen.blit(z_title, (col1_x + 14, cy + 10))
        self.screen.blit(z_desc1, (col1_x + 14, cy + 28))
        self.screen.blit(z_desc2, (col1_x + 14, cy + 44))

        # ----------------------------------------------------------------------
        # Field Definitions for Right Column (Q-Learning Hyperparameters)
        # ----------------------------------------------------------------------
        right_fields = [
            ("learning_rate", "Learning Rate (α)", "Step size for Q-value updates (0.01 - 1.0)", False, 0.01, 0.001, 1.0, 2),
            ("discount_factor", "Discount Factor (γ)", "Importance of delayed future rewards (0.0 - 1.0)", False, 0.05, 0.0, 1.0, 2),
            ("initial_epsilon", "Initial Epsilon (ε₀)", "Starting random exploration probability (0.0 - 1.0)", False, 0.05, 0.0, 1.0, 2),
            ("min_epsilon", "Minimum Epsilon (ε_min)", "Exploration floor for late training (0.0 - 1.0)", False, 0.01, 0.0, 1.0, 2),
            ("epsilon_decay", "Epsilon Decay Factor", "Multiplicative decay per episode (default: 0.995)", False, 0.005, 0.5, 0.9999, 3),
        ]

        for idx, (key, label, subtitle, is_int, step, min_v, max_v, dec) in enumerate(right_fields):
            cy = start_y + idx * (row_h + gap_y)
            c_rect = pygame.Rect(col2_x, cy, col2_w, row_h)
            pygame.draw.rect(self.screen, (255, 255, 255), c_rect, border_radius=8)
            pygame.draw.rect(self.screen, (226, 232, 240), c_rect, 1, border_radius=8)
            pygame.draw.rect(self.screen, (147, 51, 234), (col2_x, cy, 4, row_h), border_top_left_radius=8, border_bottom_left_radius=8)

            # Text Labels
            l_surf = self.font_body_bold.render(label, True, (15, 23, 42))
            s_surf = self.font_stat_sub.render(subtitle, True, (100, 116, 139))
            self.screen.blit(l_surf, (col2_x + 14, cy + 12))
            self.screen.blit(s_surf, (col2_x + 14, cy + 34))

            # Stepper & Value Controls
            btn_w = 30
            btn_h = 32
            val_w = 90
            ctrl_y = cy + (row_h - btn_h) // 2
            minus_x = col2_x + col2_w - 14 - (btn_w * 2 + val_w + 8)
            val_x = minus_x + btn_w + 4
            plus_x = val_x + val_w + 4

            minus_rect = pygame.Rect(minus_x, ctrl_y, btn_w, btn_h)
            val_rect = pygame.Rect(val_x, ctrl_y, val_w, btn_h)
            plus_rect = pygame.Rect(plus_x, ctrl_y, btn_w, btn_h)

            self.settings_rects[f"{key}_minus"] = (minus_rect, key, "minus", step, min_v, max_v, dec)
            self.settings_rects[f"{key}_plus"] = (plus_rect, key, "plus", step, min_v, max_v, dec)
            self.settings_rects[f"{key}_val"] = (val_rect, key, "val", dec)

            # Draw Minus Button
            m_hover = minus_rect.collidepoint(mouse_pos)
            pygame.draw.rect(self.screen, (226, 232, 240) if m_hover else (241, 245, 249), minus_rect, border_radius=6)
            pygame.draw.rect(self.screen, (203, 213, 225), minus_rect, 1, border_radius=6)
            m_text = self.font_btn.render("-", True, (30, 41, 59))
            self.screen.blit(m_text, (minus_rect.centerx - m_text.get_width() // 2, minus_rect.centery - m_text.get_height() // 2 - 1))

            # Draw Value Box
            is_active = (self.settings_active_field == key)
            pygame.draw.rect(self.screen, (255, 255, 255), val_rect, border_radius=6)
            if is_active:
                pygame.draw.rect(self.screen, (147, 51, 234), val_rect, 2, border_radius=6)
            else:
                pygame.draw.rect(self.screen, (203, 213, 225), val_rect, 1, border_radius=6)

            cur_val = self.settings_data.get(key, min_v)
            if is_active:
                display_str = self.settings_text_buffer + ("|" if (pygame.time.get_ticks() // 500) % 2 == 0 else "")
            else:
                display_str = f"{cur_val:.{dec}f}"

            v_surf = self.font_val.render(display_str, True, (15, 23, 42) if not is_active else (147, 51, 234))
            self.screen.blit(v_surf, (val_rect.centerx - v_surf.get_width() // 2, val_rect.centery - v_surf.get_height() // 2))

            # Draw Plus Button
            p_hover = plus_rect.collidepoint(mouse_pos)
            pygame.draw.rect(self.screen, (226, 232, 240) if p_hover else (241, 245, 249), plus_rect, border_radius=6)
            pygame.draw.rect(self.screen, (203, 213, 225), plus_rect, 1, border_radius=6)
            p_text = self.font_btn.render("+", True, (30, 41, 59))
            self.screen.blit(p_text, (plus_rect.centerx - p_text.get_width() // 2, plus_rect.centery - p_text.get_height() // 2 - 1))

        # ----------------------------------------------------------------------
        # 3. Status Notification Banner
        # ----------------------------------------------------------------------
        stat_y = start_y + 5 * (row_h + gap_y) - 2
        stat_h = 36
        stat_rect = pygame.Rect(col1_x, stat_y, col1_w * 2 + 30, stat_h)

        status_type, status_text = self.settings_status
        if status_type == "success":
            bg_col = (240, 253, 244)
            bd_col = (187, 247, 208)
            tx_col = (22, 101, 52)
            prefix = "[✓] "
        elif status_type == "error":
            bg_col = (254, 242, 242)
            bd_col = (254, 202, 202)
            tx_col = (185, 28, 28)
            prefix = "[✗] "
        elif status_type == "info":
            bg_col = (239, 246, 255)
            bd_col = (191, 219, 254)
            tx_col = (29, 78, 216)
            prefix = "[ℹ] "
        else:
            bg_col = (248, 250, 252)
            bd_col = (226, 232, 240)
            tx_col = (71, 85, 105)
            prefix = "[ℹ] "

        pygame.draw.rect(self.screen, bg_col, stat_rect, border_radius=6)
        pygame.draw.rect(self.screen, bd_col, stat_rect, 1, border_radius=6)
        msg_surf = self.font_stat_sub.render(prefix + status_text, True, tx_col)
        self.screen.blit(msg_surf, (stat_rect.centerx - msg_surf.get_width() // 2, stat_rect.centery - msg_surf.get_height() // 2))

        # ----------------------------------------------------------------------
        # 4. Safety Guardrails & Viva Rule Callout Card
        # ----------------------------------------------------------------------
        callout_y = stat_y + stat_h + 8
        callout_h = 60
        callout_rect = pygame.Rect(col1_x, callout_y, col1_w * 2 + 30, callout_h)
        pygame.draw.rect(self.screen, (245, 243, 255), callout_rect, border_radius=6)
        pygame.draw.rect(self.screen, (233, 213, 255), callout_rect, 1, border_radius=6)

        c_title = self.font_card_title.render("ALGORITHMIC GUARDRAILS & PERSISTENCE (VIVA TAKEAWAY):", True, (109, 40, 217))
        c_line1 = self.font_stat_sub.render(
            "• Negative learning rate, discount factor > 1, negative episodes, and zero actions are strictly prohibited.",
            True,
            (76, 29, 149),
        )
        c_line2 = self.font_stat_sub.render(
            "• Settings are saved to 'config.json' and dynamically update active campus environment and agent instances.",
            True,
            (76, 29, 149),
        )
        self.screen.blit(c_title, (col1_x + 14, callout_y + 8))
        self.screen.blit(c_line1, (col1_x + 14, callout_y + 24))
        self.screen.blit(c_line2, (col1_x + 14, callout_y + 40))

        # ----------------------------------------------------------------------
        # 5. Fixed Bottom Action Bar
        # ----------------------------------------------------------------------
        bar_y = 665
        pygame.draw.line(self.screen, (226, 232, 240), (0, bar_y), (config.SCREEN_WIDTH, bar_y), 1)

        self.btn_save_settings.update(mouse_pos)
        self.btn_save_settings.draw(self.screen)

        self.btn_reset_settings.update(mouse_pos)
        self.btn_reset_settings.draw(self.screen)

        self.btn_back_settings.update(mouse_pos)
        self.btn_back_settings.draw(self.screen)

    # --------------------------------------------------------------------------
    # Main Render & Event Dispatcher
    # --------------------------------------------------------------------------

    def draw(self) -> None:
        """Render the currently active menu or sub-screen."""
        if self.sub_screen == "MAIN":
            self._draw_main_menu()
        elif self.sub_screen == "RESULTS":
            self._draw_results_screen()
        elif self.sub_screen == "RL_WORKS":
            self._draw_rl_guide_screen()
        elif self.sub_screen == "SETTINGS":
            self._draw_settings_screen()

        pygame.display.flip()

    def handle_input(self) -> Optional[int]:
        """
        Handle mouse and keyboard events.

        Returns:
            option_code: 1 to 6 if an executable action is selected, or None.
        """
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 6  # Exit

            # ------------------------------------------------------------------
            # Sub-Screen Navigation: [4] RESULTS Dashboard
            # ------------------------------------------------------------------
            if self.sub_screen == "RESULTS":
                # Fallback mode when no results data exists
                if self.results_metrics is None:
                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_1, pygame.K_t, pygame.K_RETURN):
                            self.sub_screen = "MAIN"
                            return 1  # Trigger AI training mode
                        elif event.key in (pygame.K_ESCAPE, pygame.K_m, pygame.K_SPACE):
                            self.sub_screen = "MAIN"
                            return None

                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if self.btn_no_results_train.is_clicked(mouse_pos, event):
                            self.sub_screen = "MAIN"
                            return 1
                        elif self.btn_no_results_back.is_clicked(mouse_pos, event):
                            self.sub_screen = "MAIN"
                            return None
                    continue

                # Active Results Dashboard Mode
                # 1. Mouse Wheel Scrolling
                if event.type == pygame.MOUSEWHEEL:
                    self.results_scroll_pos -= event.y * 45
                    self.results_scroll_pos = max(0.0, min(self.results_max_scroll, self.results_scroll_pos))
                    continue

                # 2. Mouse Drag & Click on Scrollbar and Action Buttons
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Check scrollbar track click
                    if 1060 <= mouse_pos[0] <= 1085 and 90 <= mouse_pos[1] <= 665:
                        rel_y = mouse_pos[1] - 90
                        ratio = max(0.0, min(1.0, rel_y / 575.0))
                        self.results_scroll_pos = ratio * self.results_max_scroll
                        self.is_dragging_scrollbar = True
                        continue

                    # Check Action Buttons
                    if self.btn_open_folder.is_clicked(mouse_pos, event):
                        self._open_results_folder()
                        continue
                    elif self.btn_results_back.is_clicked(mouse_pos, event):
                        self.sub_screen = "MAIN"
                        return None

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.is_dragging_scrollbar = False

                if event.type == pygame.MOUSEMOTION and self.is_dragging_scrollbar:
                    rel_y = mouse_pos[1] - 90
                    ratio = max(0.0, min(1.0, rel_y / 575.0))
                    self.results_scroll_pos = ratio * self.results_max_scroll
                    self.results_scroll_pos = max(0.0, min(self.results_max_scroll, self.results_scroll_pos))
                    continue

                # 3. Keyboard Scrolling & Navigation
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_UP, pygame.K_w):
                        self.results_scroll_pos = max(0.0, self.results_scroll_pos - 40)
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        self.results_scroll_pos = min(self.results_max_scroll, self.results_scroll_pos + 40)
                    elif event.key == pygame.K_PAGEUP:
                        self.results_scroll_pos = max(0.0, self.results_scroll_pos - 250)
                    elif event.key in (pygame.K_PAGEDOWN, pygame.K_SPACE):
                        self.results_scroll_pos = min(self.results_max_scroll, self.results_scroll_pos + 250)
                    elif event.key == pygame.K_HOME:
                        self.results_scroll_pos = 0.0
                    elif event.key == pygame.K_END:
                        self.results_scroll_pos = self.results_max_scroll
                    elif event.key in (pygame.K_o, pygame.K_f):
                        self._open_results_folder()
                    elif event.key in (pygame.K_ESCAPE, pygame.K_m):
                        self.sub_screen = "MAIN"
                        return None
                continue

            # ------------------------------------------------------------------
            # Sub-Screen Navigation: [5] RL_WORKS
            # ------------------------------------------------------------------
            if self.sub_screen == "RL_WORKS":
                # 1. Mouse Wheel Scrolling
                if event.type == pygame.MOUSEWHEEL:
                    self.rl_scroll_pos -= event.y * 45
                    self.rl_scroll_pos = max(0.0, min(self.rl_max_scroll, self.rl_scroll_pos))
                    continue

                # 2. Mouse Drag & Click on Scrollbar and Back Button
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # Check scrollbar track click (track_x = 1068, track_w = 8, track_y = 90..665)
                    if 1060 <= mouse_pos[0] <= 1085 and 90 <= mouse_pos[1] <= 665:
                        rel_y = mouse_pos[1] - 90
                        ratio = max(0.0, min(1.0, rel_y / 575.0))
                        self.rl_scroll_pos = ratio * self.rl_max_scroll
                        self.is_dragging_rl_scrollbar = True
                        continue

                    # Check Back Button
                    if self.btn_rl_back.is_clicked(mouse_pos, event):
                        self.sub_screen = "MAIN"
                        return None

                if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.is_dragging_rl_scrollbar = False

                if event.type == pygame.MOUSEMOTION and self.is_dragging_rl_scrollbar:
                    rel_y = mouse_pos[1] - 90
                    ratio = max(0.0, min(1.0, rel_y / 575.0))
                    self.rl_scroll_pos = ratio * self.rl_max_scroll
                    self.rl_scroll_pos = max(0.0, min(self.rl_max_scroll, self.rl_scroll_pos))
                    continue

                # 3. Keyboard Scrolling & Navigation
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_UP, pygame.K_w):
                        self.rl_scroll_pos = max(0.0, self.rl_scroll_pos - 40)
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        self.rl_scroll_pos = min(self.rl_max_scroll, self.rl_scroll_pos + 40)
                    elif event.key == pygame.K_PAGEUP:
                        self.rl_scroll_pos = max(0.0, self.rl_scroll_pos - 250)
                    elif event.key in (pygame.K_PAGEDOWN, pygame.K_SPACE):
                        self.rl_scroll_pos = min(self.rl_max_scroll, self.rl_scroll_pos + 250)
                    elif event.key == pygame.K_HOME:
                        self.rl_scroll_pos = 0.0
                    elif event.key == pygame.K_END:
                        self.rl_scroll_pos = self.rl_max_scroll
                    elif event.key in (pygame.K_ESCAPE, pygame.K_m, pygame.K_RETURN):
                        self.sub_screen = "MAIN"
                        return None
                continue

            # ------------------------------------------------------------------
            # Sub-Screen Navigation: [6] SETTINGS
            # ------------------------------------------------------------------
            if self.sub_screen == "SETTINGS":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # 1. Action Buttons
                    if self.btn_save_settings.is_clicked(mouse_pos, event):
                        self._save_settings_from_form()
                        continue
                    elif self.btn_reset_settings.is_clicked(mouse_pos, event):
                        self._reset_settings_to_defaults()
                        continue
                    elif self.btn_back_settings.is_clicked(mouse_pos, event):
                        self._commit_active_text_field()
                        self.sub_screen = "MAIN"
                        return None

                    # 2. Check Speed Segmented Buttons
                    clicked_speed = False
                    for sp in ("SLOW", "NORMAL", "FAST"):
                        sp_key = f"speed_{sp}"
                        if sp_key in self.settings_rects:
                            sp_rect = self.settings_rects[sp_key][0]
                            if sp_rect.collidepoint(mouse_pos):
                                self._commit_active_text_field()
                                self.settings_data["training_speed"] = sp
                                self.settings_status = ("normal", f"Training speed set to {sp}")
                                clicked_speed = True
                                break
                    if clicked_speed:
                        continue

                    # 3. Check Stepper Buttons and Value Box Clicks
                    clicked_field = False
                    for rect_key, data in self.settings_rects.items():
                        if rect_key.startswith("speed_"):
                            continue
                        rect = data[0]
                        if rect.collidepoint(mouse_pos):
                            clicked_field = True
                            action = data[2]
                            field_key = data[1]

                            if action == "val":
                                # Clicked on a value box to type
                                self._commit_active_text_field()
                                self.settings_active_field = field_key
                                cur = self.settings_data.get(field_key, 0)
                                self.settings_text_buffer = str(cur)
                            elif action == "minus":
                                self._commit_active_text_field()
                                step, min_v, max_v, dec = data[3], data[4], data[5], data[6]
                                cur = self.settings_data.get(field_key, min_v)
                                if dec == 0:
                                    new_v = max(min_v, int(cur - step))
                                else:
                                    new_v = max(min_v, round(cur - step, dec))
                                self.settings_data[field_key] = new_v
                                self.settings_status = ("normal", f"Adjusted {field_key} to {new_v}")
                            elif action == "plus":
                                self._commit_active_text_field()
                                step, min_v, max_v, dec = data[3], data[4], data[5], data[6]
                                cur = self.settings_data.get(field_key, min_v)
                                if dec == 0:
                                    new_v = min(max_v, int(cur + step))
                                else:
                                    new_v = min(max_v, round(cur + step, dec))
                                self.settings_data[field_key] = new_v
                                self.settings_status = ("normal", f"Adjusted {field_key} to {new_v}")
                            break

                    if not clicked_field and self.settings_active_field:
                        # Clicked elsewhere on screen, commit active text field
                        self._commit_active_text_field()
                    continue

                if event.type == pygame.KEYDOWN:
                    if self.settings_active_field:
                        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            self._commit_active_text_field()
                        elif event.key == pygame.K_ESCAPE:
                            self.settings_active_field = None
                            self.settings_text_buffer = ""
                        elif event.key == pygame.K_BACKSPACE:
                            self.settings_text_buffer = self.settings_text_buffer[:-1]
                        elif event.unicode and (event.unicode.isdigit() or event.unicode in (".", "-")):
                            if len(self.settings_text_buffer) < 10:
                                self.settings_text_buffer += event.unicode
                    else:
                        if event.key in (pygame.K_ESCAPE, pygame.K_m, pygame.K_b):
                            self.sub_screen = "MAIN"
                            return None
                        elif event.key == pygame.K_s:
                            self._save_settings_from_form()
                        elif event.key == pygame.K_r:
                            self._reset_settings_to_defaults()
                        elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                            self._save_settings_from_form()
                continue

            # ------------------------------------------------------------------
            # Main Menu Mouse Interaction
            # ------------------------------------------------------------------
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for idx, btn in enumerate(self.buttons):
                    if btn.is_clicked(mouse_pos, event):
                        code = self.option_labels[idx][2]
                        return self._process_selection(code)

            # ------------------------------------------------------------------
            # Main Menu Keyboard Interaction
            # ------------------------------------------------------------------
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_UP, pygame.K_w):
                    self.selected_index = (self.selected_index - 1) % len(self.buttons)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    self.selected_index = (self.selected_index + 1) % len(self.buttons)
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    code = self.option_labels[self.selected_index][2]
                    return self._process_selection(code)
                elif event.key == pygame.K_1:
                    return self._process_selection(1)
                elif event.key == pygame.K_2:
                    return self._process_selection(2)
                elif event.key == pygame.K_3:
                    return self._process_selection(3)
                elif event.key == pygame.K_4:
                    return self._process_selection(4)
                elif event.key == pygame.K_5:
                    return self._process_selection(5)
                elif event.key == pygame.K_6:
                    return self._process_selection(6)
                elif event.key in (pygame.K_7, pygame.K_s):
                    return self._process_selection(7)
                elif event.key in (pygame.K_8, pygame.K_ESCAPE):
                    return self._process_selection(8)

        return None

    def _process_selection(self, code: int) -> Optional[int]:
        """Route internal menu views or dispatch external action codes."""
        if code == 5:
            # Internal modal: [5] View Results Dashboard
            self.sub_screen = "RESULTS"
            self.results_scroll_pos = 0.0
            self._load_results_data(force_reload=True)
            self._load_results_plots(force_reload=True)
            return None
        elif code == 6:
            # Internal modal: [6] Learn RL ("How RL Works")
            self.sub_screen = "RL_WORKS"
            self.rl_scroll_pos = 0.0
            return None
        elif code == 7:
            # Internal modal: [7] Settings
            self.sub_screen = "SETTINGS"
            self._load_settings_into_form()
            return None
        # Return executable codes [1: Train AI, 2: Watch AI, 3: Play Game, 4: Compare Human vs AI, 8: Exit]
        return code
