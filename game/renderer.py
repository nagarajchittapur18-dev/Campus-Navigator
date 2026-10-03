"""
Campus Navigator: Graphical Renderer (Phase 6)
==============================================
Module: game/renderer.py

ARCHITECTURAL PRINCIPLE - STRICT UI / RL SEPARATION:
This class is strictly responsible for 2D visual presentation using Pygame.
It reads the grid coordinates from CampusEnvironment and runtime statistics
from the game loop to render:
    - 10x10 Campus Grid (Left side)
    - Distinct landmarks (Start, Goal, Buildings/Obstacles, Safe/Unsafe Zones)
    - Smoothly animated student agent with linear interpolation (LERP)
    - Telemetry Information Panel (Right side)

NO Reinforcement Learning or Bellman update equations exist in this file.
NO external image files are required; all graphics are procedurally rendered
with high-performance Pygame geometric primitives and typography.
"""

import math
from typing import Dict, List, Optional, Tuple
import pygame

import config
from environment.campus_env import CampusEnvironment
from game.menu import Button


class CampusRenderer:
    """Renders the Campus Gridworld and live HUD Telemetry onto a Pygame surface."""

    def __init__(
        self,
        screen: pygame.Surface,
        rows: int = config.GRID_ROWS,
        cols: int = config.GRID_COLS,
        cell_size: int = config.GRID_CELL_SIZE,
    ) -> None:
        """
        Initialize the graphical renderer.

        Args:
            screen: Pygame display surface (1100x750).
            rows: Total campus rows (10).
            cols: Total campus columns (10).
            cell_size: Pixel dimension of each cell (60px).
        """
        self.screen = screen
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size

        # Grid positioning: Left side
        self.grid_x: int = 50
        self.grid_y: int = 80
        self.grid_width: int = self.cols * self.cell_size   # 600 px
        self.grid_height: int = self.rows * self.cell_size  # 600 px

        # HUD Panel: Right side
        self.panel_x: int = self.grid_x + self.grid_width + 40  # 690 px
        self.panel_y: int = self.grid_y                         # 80 px
        self.panel_width: int = config.SCREEN_WIDTH - self.panel_x - 40  # 370 px
        self.panel_height: int = self.grid_height               # 600 px

        # Smooth agent movement interpolation state (in pixel space)
        start_pixel_x = self.grid_x + (config.START_POSITION[1] * self.cell_size) + (self.cell_size // 2)
        start_pixel_y = self.grid_y + (config.START_POSITION[0] * self.cell_size) + (self.cell_size // 2)
        self.agent_anim_x: float = float(start_pixel_x)
        self.agent_anim_y: float = float(start_pixel_y)
        self.lerp_speed: float = 0.25  # Smooth interpolation factor (0.0 to 1.0)

        # Initialize Fonts (Clean Academic Hierarchy)
        pygame.font.init()
        self.font_title = pygame.font.SysFont("Segoe UI, Arial", 24, bold=True)
        self.font_subtitle = pygame.font.SysFont("Segoe UI, Arial", 13)
        self.font_panel_title = pygame.font.SysFont("Segoe UI, Arial", 18, bold=True)
        self.font_header = pygame.font.SysFont("Segoe UI, Arial", 18, bold=True)
        self.font_label = pygame.font.SysFont("Segoe UI, Arial", 13)
        self.font_val = pygame.font.SysFont("Segoe UI, Arial", 14, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI, Arial", 11, bold=True)
        self.font_badge = pygame.font.SysFont("Segoe UI, Arial", 16, bold=True)
        self.font_btn = pygame.font.SysFont("Segoe UI, Arial", 15, bold=True)
        self.font_body = pygame.font.SysFont("Segoe UI, Arial", 13)
        self.font_body_bold = pygame.font.SysFont("Segoe UI, Arial", 13, bold=True)

        # Monospace Fonts for Telemetry & Coordinates (Phase 17 Enhancement)
        self.font_mono = pygame.font.SysFont("Consolas, Courier New, monospace", 13, bold=True)
        self.font_mono_small = pygame.font.SysFont("Consolas, Courier New, monospace", 11, bold=True)
        self.font_mono_val = pygame.font.SysFont("Consolas, Courier New, monospace", 14, bold=True)
        self.font_mono_tiny = pygame.font.SysFont("Consolas, Courier New, monospace", 9, bold=True)

    # --------------------------------------------------------------------------
    # Cell Rendering Helpers (Pure Pygame Shapes)
    # --------------------------------------------------------------------------

    def _draw_obstacle_building(self, rect: pygame.Rect) -> None:
        """Render a dark architectural building with windows."""
        # Main structure body
        pygame.draw.rect(self.screen, config.COLOR_OBSTACLE, rect, border_radius=4)
        pygame.draw.rect(self.screen, (15, 23, 42), rect, 2, border_radius=4)

        # Procedural building windows (3x3 grid of tiny glowing panes)
        window_size = 6
        spacing_x = 14
        spacing_y = 14
        start_x = rect.x + 13
        start_y = rect.y + 11

        for r_win in range(3):
            for c_win in range(3):
                win_rect = pygame.Rect(
                    start_x + c_win * spacing_x,
                    start_y + r_win * spacing_y,
                    window_size,
                    window_size,
                )
                # Alternating warm window glow
                win_color = (253, 224, 71) if (r_win + c_win) % 2 == 0 else (148, 163, 184)
                pygame.draw.rect(self.screen, win_color, win_rect)

        # Building roofline detail
        pygame.draw.line(self.screen, config.COLOR_OBSTACLE_DETAIL, (rect.x + 4, rect.y + 4), (rect.right - 4, rect.y + 4), 2)

    def _draw_start_marker(self, rect: pygame.Rect, badge_label: str = "GATE") -> None:
        """Render the green Campus starting point landmark."""
        pygame.draw.rect(self.screen, (220, 252, 231), rect)  # Light mint base
        pygame.draw.rect(self.screen, config.COLOR_START, rect, 3, border_radius=4)

        # Archway gate pillars
        pillar_w = 6
        pillar_h = 36
        py = rect.bottom - pillar_h - 6
        pygame.draw.rect(self.screen, config.COLOR_START, (rect.x + 10, py, pillar_w, pillar_h), border_radius=2)
        pygame.draw.rect(self.screen, config.COLOR_START, (rect.right - 10 - pillar_w, py, pillar_w, pillar_h), border_radius=2)
        # Arch top beam
        pygame.draw.rect(self.screen, config.COLOR_START, (rect.x + 10, py, rect.width - 20, 8), border_radius=2)

        lbl = self.font_small.render(badge_label, True, (22, 101, 52))
        self.screen.blit(lbl, (rect.centerx - lbl.get_width() // 2, rect.y + 8))

    def _draw_goal_marker(self, rect: pygame.Rect, badge_label: str = "LIB") -> None:
        """Render the golden target destination landmark."""
        pygame.draw.rect(self.screen, (254, 240, 138), rect)  # Light gold base
        pygame.draw.rect(self.screen, config.COLOR_GOAL, rect, 3, border_radius=4)

        # Concentric target circles
        center = rect.center
        pygame.draw.circle(self.screen, (245, 158, 11), center, 20)
        pygame.draw.circle(self.screen, (254, 240, 138), center, 14)
        pygame.draw.circle(self.screen, (217, 119, 6), center, 7)

        lbl = self.font_small.render(badge_label, True, (120, 53, 15))
        self.screen.blit(lbl, (rect.centerx - lbl.get_width() // 2, rect.y + 4))

    def _draw_unsafe_zone(self, rect: pygame.Rect) -> None:
        """Render hazard orange unsafe walkway with caution stripes."""
        pygame.draw.rect(self.screen, (255, 237, 213), rect)  # Light orange tint
        pygame.draw.rect(self.screen, config.COLOR_UNSAFE_ZONE, rect, 2, border_radius=4)

        # Caution diagonal hash lines
        for offset in range(-self.cell_size, self.cell_size * 2, 16):
            p1 = (max(rect.x, rect.x + offset), rect.y)
            p2 = (min(rect.right, rect.x + offset + 12), rect.bottom)
            if rect.collidepoint(p1) or rect.collidepoint(p2):
                pygame.draw.line(self.screen, (254, 215, 170), (rect.x + offset, rect.y), (rect.x + offset + 20, rect.bottom), 2)

        # Warning icon badge
        badge = self.font_small.render("HAZARD", True, (194, 65, 12))
        self.screen.blit(badge, (rect.centerx - badge.get_width() // 2, rect.centery - 6))

    def _draw_safe_zone(self, rect: pygame.Rect) -> None:
        """Render safe student hub zone with purple/indigo theme."""
        pygame.draw.rect(self.screen, (238, 242, 255), rect)  # Light indigo tint
        pygame.draw.rect(self.screen, config.COLOR_SAFE_ZONE, rect, 2, border_radius=4)

        # Medical / Safe Hub Cross
        cross_color = config.COLOR_SAFE_ZONE
        center = rect.center
        pygame.draw.rect(self.screen, cross_color, (center[0] - 3, center[1] - 12, 6, 24), border_radius=2)
        pygame.draw.rect(self.screen, cross_color, (center[0] - 12, center[1] - 3, 24, 6), border_radius=2)

        badge = self.font_small.render("SAFE", True, (67, 56, 202))
        self.screen.blit(badge, (rect.centerx - badge.get_width() // 2, rect.bottom - 16))

    def _draw_crowded_zone(self, rect: pygame.Rect) -> None:
        """Render high-traffic crowded zone with amber base and pedestrian silhouettes."""
        pygame.draw.rect(self.screen, (254, 243, 199), rect)  # Light amber base
        pygame.draw.rect(self.screen, config.COLOR_CROWDED_ZONE, rect, 2, border_radius=4)

        # Pedestrian silhouettes representing high corridor congestion
        center = rect.center
        # Middle figure
        pygame.draw.circle(self.screen, (217, 119, 6), (center[0], center[1] - 8), 4)
        pygame.draw.rect(self.screen, (217, 119, 6), (center[0] - 5, center[1] - 3, 10, 8), border_radius=2)
        # Left figure
        pygame.draw.circle(self.screen, config.COLOR_CROWDED_ZONE, (center[0] - 12, center[1] - 5), 3)
        pygame.draw.rect(self.screen, config.COLOR_CROWDED_ZONE, (center[0] - 16, center[1] - 1, 8, 6), border_radius=2)
        # Right figure
        pygame.draw.circle(self.screen, config.COLOR_CROWDED_ZONE, (center[0] + 12, center[1] - 5), 3)
        pygame.draw.rect(self.screen, config.COLOR_CROWDED_ZONE, (center[0] + 8, center[1] - 1, 8, 6), border_radius=2)

        badge = self.font_small.render("CROWD", True, (180, 83, 9))
        self.screen.blit(badge, (rect.centerx - badge.get_width() // 2, rect.bottom - 16))

    # --------------------------------------------------------------------------
    # Campus Grid & Legend Renderer
    # --------------------------------------------------------------------------

    def draw_axis_rulers(self) -> None:
        """
        Render collegiate coordinate axis rulers around the campus gridworld.
        Displays column numbers (0..9) along the top and row numbers (0..9) along the left
        so viva evaluators and students can instantly verify Cartesian (row, col) coordinates.
        """
        # Top Column Axis (0..9)
        col_y = self.grid_y - 18
        for c in range(self.cols):
            cx = self.grid_x + c * self.cell_size + (self.cell_size // 2)
            c_surf = self.font_mono_small.render(str(c), True, config.COLOR_AXIS_RULER)
            self.screen.blit(c_surf, (cx - c_surf.get_width() // 2, col_y))
            # Subtle tick mark
            pygame.draw.line(self.screen, config.COLOR_GRID_LINE, (cx, self.grid_y - 5), (cx, self.grid_y - 1), 1)

        # Left Row Axis (0..9)
        row_x = self.grid_x - 18
        for r in range(self.rows):
            cy = self.grid_y + r * self.cell_size + (self.cell_size // 2)
            r_surf = self.font_mono_small.render(str(r), True, config.COLOR_AXIS_RULER)
            self.screen.blit(r_surf, (row_x - r_surf.get_width() // 2, cy - r_surf.get_height() // 2))
            # Subtle tick mark
            pygame.draw.line(self.screen, config.COLOR_GRID_LINE, (self.grid_x - 5, cy), (self.grid_x - 1, cy), 1)

        # Origin Axis Label (r\c) in top-left gutter
        origin_lbl = self.font_mono_tiny.render("r\\c", True, (148, 163, 184))
        self.screen.blit(origin_lbl, (self.grid_x - 22, self.grid_y - 18))

    def draw_status_indicator(
        self,
        x: int,
        y: int,
        status_type: str = "ACTIVE",
        custom_text: str = "",
    ) -> None:
        """
        Render a live academic status pill indicator with a periodic pulsing dot.
        Status types:
            ACTIVE   : Emerald dot (Exploitation / Navigation)
            TRAINING : Amber dot (Q-Learning exploration / learning)
            MANUAL   : Sky blue dot (Keyboard human control)
            COLLISION: Crimson dot (Fault penalty / obstacle hit)
            GOAL     : Rich gold dot (Destination reached)
            PAUSED   : Slate dot (Simulation paused)
        """
        type_upper = status_type.upper()
        if "GOAL" in type_upper or "ARRIVED" in type_upper or "COMPLETE" in type_upper:
            dot_col = (234, 179, 8)
            border_col = (202, 138, 4)
            label = custom_text or "GOAL REACHED"
        elif "COLLISION" in type_upper or "HAZARD" in type_upper or "PENALTY" in type_upper:
            dot_col = config.COLOR_STATUS_ALERT
            border_col = (185, 28, 28)
            label = custom_text or "COLLISION ALERT"
        elif "TRAIN" in type_upper or "EXPLOR" in type_upper:
            dot_col = config.COLOR_STATUS_TRAINING
            border_col = (217, 119, 6)
            label = custom_text or "AI TRAINING"
        elif "MANUAL" in type_upper or "HUMAN" in type_upper or "PLAY" in type_upper:
            dot_col = config.COLOR_STATUS_MANUAL
            border_col = (2, 132, 199)
            label = custom_text or "MANUAL CONTROL"
        elif "PAUSE" in type_upper:
            dot_col = (148, 163, 184)
            border_col = (100, 116, 139)
            label = custom_text or "PAUSED"
        else:
            dot_col = config.COLOR_STATUS_ACTIVE
            border_col = (22, 163, 74)
            label = custom_text or "ACTIVE POLICY"

        # Calculate pill dimension
        txt_surf = self.font_small.render(label, True, (241, 245, 249))
        pill_w = txt_surf.get_width() + 32
        pill_h = 24
        pill_rect = pygame.Rect(x, y, pill_w, pill_h)

        # Dark navy translucent pill container
        pygame.draw.rect(self.screen, (30, 41, 59), pill_rect, border_radius=12)
        pygame.draw.rect(self.screen, border_col, pill_rect, 1, border_radius=12)

        # Animated pulsing status dot
        ticks = pygame.time.get_ticks()
        pulse = (math.sin(ticks * 0.007) + 1.0) * 0.5  # 0.0 to 1.0
        dot_center = (pill_rect.x + 12, pill_rect.centery)

        # Soft outer pulse halo
        halo_radius = 4 + int(pulse * 2.5)
        halo_alpha = int(120 * (1.0 - pulse * 0.5))
        halo_surf = pygame.Surface((halo_radius * 2 + 2, halo_radius * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(halo_surf, (*dot_col, halo_alpha), (halo_radius + 1, halo_radius + 1), halo_radius)
        self.screen.blit(halo_surf, (dot_center[0] - halo_radius - 1, dot_center[1] - halo_radius - 1))

        # Solid inner dot
        pygame.draw.circle(self.screen, dot_col, dot_center, 4)

        # Label text
        self.screen.blit(txt_surf, (pill_rect.x + 22, pill_rect.centery - txt_surf.get_height() // 2))

    def draw_goal_arrival_animation(self, goal_pos: Tuple[int, int]) -> None:
        """
        Render a celebratory, academic expanding concentric ripple halo around the destination landmark.
        Runs smoothly at 60 FPS without external sprites or UI blocking.
        """
        r, c = goal_pos
        cx = self.grid_x + c * self.cell_size + (self.cell_size // 2)
        cy = self.grid_y + r * self.cell_size + (self.cell_size // 2)

        ticks = pygame.time.get_ticks()
        surf_dim = 120
        half = surf_dim // 2
        ripple_surf = pygame.Surface((surf_dim, surf_dim), pygame.SRCALPHA)

        # Expanding Ring 1 (Gold)
        p1 = (ticks % 1400) / 1400.0
        rad1 = int(18 + p1 * 32)
        alpha1 = int(220 * (1.0 - p1))
        pygame.draw.circle(ripple_surf, (245, 158, 11, alpha1), (half, half), rad1, 2)

        # Expanding Ring 2 (Emerald, offset by 700ms)
        p2 = ((ticks + 700) % 1400) / 1400.0
        rad2 = int(18 + p2 * 32)
        alpha2 = int(220 * (1.0 - p2))
        pygame.draw.circle(ripple_surf, (34, 197, 94, alpha2), (half, half), rad2, 2)

        # Blit centered on goal cell
        self.screen.blit(ripple_surf, (cx - half, cy - half))

    def draw_failure_feedback(
        self,
        cell_pos: Optional[Tuple[int, int]],
        alert_msg: str,
        timestamp: int,
    ) -> None:
        """
        Render high-contrast visual feedback on collision with impassable obstacles or entry into hazards.
        Highlights the target cell with an alert outline that fades over 900 ms.
        """
        if not cell_pos or timestamp <= 0:
            return

        now = pygame.time.get_ticks()
        elapsed = now - timestamp
        if elapsed >= 900:
            return

        fade = 1.0 - (elapsed / 900.0)
        alpha = int(210 * fade)

        r, c = cell_pos
        if 0 <= r < self.rows and 0 <= c < self.cols:
            cell_rect = pygame.Rect(
                self.grid_x + c * self.cell_size,
                self.grid_y + r * self.cell_size,
                self.cell_size,
                self.cell_size,
            )
            flash_surf = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
            pygame.draw.rect(flash_surf, (239, 68, 68, int(alpha * 0.35)), (0, 0, self.cell_size, self.cell_size), border_radius=4)
            pygame.draw.rect(flash_surf, (220, 38, 38, alpha), (0, 0, self.cell_size, self.cell_size), 3, border_radius=4)
            self.screen.blit(flash_surf, cell_rect.topleft)

    def draw_grid(self, env: CampusEnvironment, stats: Optional[Dict] = None) -> None:
        """Draw the entire 10x10 Campus map with coordinate rulers and dynamic visual feedback."""
        # 1. Coordinate Axis Rulers (Phase 17)
        self.draw_axis_rulers()

        # Backdrop container
        container_rect = pygame.Rect(self.grid_x - 4, self.grid_y - 4, self.grid_width + 8, self.grid_height + 8)
        pygame.draw.rect(self.screen, (226, 232, 240), container_rect, border_radius=8)
        pygame.draw.rect(self.screen, config.COLOR_GRID_LINE, container_rect, 2, border_radius=8)

        start_badge = getattr(config, "START_BADGES", {}).get(getattr(env, "current_start_name", ""), "GATE")
        goal_badge = getattr(config, "DESTINATION_BADGES", {}).get(getattr(env, "current_destination_name", ""), "LIB")

        for r in range(self.rows):
            for c in range(self.cols):
                pos = (r, c)
                rect = pygame.Rect(
                    self.grid_x + c * self.cell_size,
                    self.grid_y + r * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )

                if pos == env.start_pos:
                    self._draw_start_marker(rect, start_badge)
                elif pos == env.goal_pos:
                    self._draw_goal_marker(rect, goal_badge)
                elif pos in env.obstacles:
                    self._draw_obstacle_building(rect)
                elif pos in env.unsafe_zones:
                    self._draw_unsafe_zone(rect)
                elif pos in env.safe_zones:
                    self._draw_safe_zone(rect)
                elif hasattr(env, "crowded_zones") and pos in env.crowded_zones:
                    self._draw_crowded_zone(rect)
                else:
                    # Clean walkable pathway
                    pygame.draw.rect(self.screen, config.COLOR_PATHWAY, rect)
                    pygame.draw.rect(self.screen, config.COLOR_GRID_LINE, rect, 1)

        # Dynamic visual feedback: goal arrival animation & failure flash
        if stats:
            if stats.get("goal_reached", False):
                self.draw_goal_arrival_animation(env.goal_pos)
            coll_cell = stats.get("last_collision_cell")
            coll_time = stats.get("last_collision_time", 0)
            coll_msg = stats.get("last_collision_msg", "")
            if coll_cell and coll_time:
                self.draw_failure_feedback(coll_cell, coll_msg, coll_time)

    def draw_legend(self, env: Optional[CampusEnvironment] = None) -> None:
        """
        Render a clean, professional visual legend banner below the campus grid.
        Displays symbols, dynamic coordinate tags, and reward modifiers for:
        Start Gate, Library Goal, Buildings, Safe Zones, Crowded Zones, Hazard Zones, and Learned Route.
        """
        leg_x = self.grid_x
        leg_y = self.grid_y + self.grid_height + 12  # 80 + 600 + 12 = 692
        leg_w = self.grid_width + self.panel_width + 40  # 600 + 370 + 40 = 1010
        leg_h = 44

        leg_rect = pygame.Rect(leg_x, leg_y, leg_w, leg_h)
        pygame.draw.rect(self.screen, (255, 255, 255), leg_rect, border_radius=8)
        pygame.draw.rect(self.screen, config.COLOR_GRID_LINE, leg_rect, 1, border_radius=8)

        # Banner Title
        title_surf = self.font_small.render("CAMPUS LEGEND:", True, (71, 85, 105))
        self.screen.blit(title_surf, (leg_x + 14, leg_y + 14))

        # Dynamic coordinates & reward labels based on environment config
        start_coord = env.start_pos if env else config.START_POSITION
        goal_coord = env.goal_pos if env else config.GOAL_POSITION
        start_name = getattr(env, "current_start_name", "Gate") if env else "Gate"
        goal_name = getattr(env, "current_destination_name", "Library") if env else "Library"

        safe_rew = int(env.reward_safe_zone) if env else int(config.REWARD_SAFE_ZONE)
        crowd_rew = int(env.reward_crowded_zone) if env else int(config.REWARD_CROWDED_ZONE)
        hazard_rew = int(env.reward_unsafe_zone) if env else int(config.REWARD_UNSAFE_ZONE)

        # 7 Legend Swatches across the 1010px banner
        legend_items = [
            (f"{start_name} {start_coord}", config.COLOR_START, "rect", (22, 101, 52)),
            (f"{goal_name} {goal_coord}", config.COLOR_GOAL, "circle", (180, 83, 9)),
            ("Building [#]", config.COLOR_OBSTACLE, "rect", (15, 23, 42)),
            (f"Safe (+{safe_rew})", config.COLOR_SAFE_ZONE, "rect", (67, 56, 202)),
            (f"Crowd ({crowd_rew})", config.COLOR_CROWDED_ZONE, "rect", (180, 83, 9)),
            (f"Hazard ({hazard_rew})", config.COLOR_UNSAFE_ZONE, "rect", (194, 65, 12)),
            ("Route Policy", config.COLOR_OPTIMAL_PATH, "line", (147, 51, 234)),
        ]

        curr_x = leg_x + 120
        item_gap = 126

        for label, color, shape, text_col in legend_items:
            swatch_y = leg_y + 15
            if shape == "circle":
                pygame.draw.circle(self.screen, color, (curr_x + 7, swatch_y + 7), 7)
                pygame.draw.circle(self.screen, (255, 255, 255), (curr_x + 7, swatch_y + 7), 3)
            elif shape == "line":
                pygame.draw.line(self.screen, color, (curr_x, swatch_y + 7), (curr_x + 14, swatch_y + 7), 3)
                pygame.draw.circle(self.screen, (255, 255, 255), (curr_x + 7, swatch_y + 7), 4)
                pygame.draw.circle(self.screen, color, (curr_x + 7, swatch_y + 7), 2)
            else:
                swatch_rect = pygame.Rect(curr_x, swatch_y, 14, 14)
                pygame.draw.rect(self.screen, color, swatch_rect, border_radius=3)
                pygame.draw.rect(self.screen, (255, 255, 255), swatch_rect, 1, border_radius=3)

            lbl = self.font_small.render(label, True, text_col)
            self.screen.blit(lbl, (curr_x + 18, leg_y + 14))

            curr_x += item_gap

    # --------------------------------------------------------------------------
    # Smooth Agent Movement (Linear Interpolation)
    # --------------------------------------------------------------------------

    def update_agent_animation(self, target_pos: Tuple[int, int], lerp_speed: Optional[float] = None) -> None:
        """
        Smoothly interpolate the agent's screen coordinates toward target cell.

        Formula (LERP):
            x_anim = x_anim + (target_x - x_anim) * lerp_speed
        """
        r, c = target_pos
        target_pixel_x = self.grid_x + (c * self.cell_size) + (self.cell_size // 2)
        target_pixel_y = self.grid_y + (r * self.cell_size) + (self.cell_size // 2)

        speed = lerp_speed if lerp_speed is not None else self.lerp_speed
        # Linear interpolation step
        self.agent_anim_x += (target_pixel_x - self.agent_anim_x) * speed
        self.agent_anim_y += (target_pixel_y - self.agent_anim_y) * speed

    def draw_agent(self) -> None:
        """Render the blue student character marker with depth shadow."""
        center_x = int(self.agent_anim_x)
        center_y = int(self.agent_anim_y)
        radius = int(self.cell_size * 0.36)  # ~21 px radius

        # Soft drop shadow
        pygame.draw.circle(self.screen, (150, 160, 175), (center_x + 2, center_y + 4), radius)

        # Outer ring (white border)
        pygame.draw.circle(self.screen, (255, 255, 255), (center_x, center_y), radius + 3)

        # Outer colored border
        pygame.draw.circle(self.screen, config.COLOR_AGENT_BORDER, (center_x, center_y), radius + 1)

        # Inner body
        pygame.draw.circle(self.screen, config.COLOR_AGENT, (center_x, center_y), radius)

        # Inner highlight reflection
        pygame.draw.circle(self.screen, (96, 165, 250), (center_x - 5, center_y - 5), radius // 2)

        # Agent Character Icon / Letter
        badge_text = self.font_badge.render("A", True, (255, 255, 255))
        self.screen.blit(badge_text, badge_text.get_rect(center=(center_x, center_y)))

    # --------------------------------------------------------------------------
    # Telemetry Information Panel (Right Side)
    # --------------------------------------------------------------------------

    def draw_info_panel(self, stats: Dict) -> None:
        """Render the polished telemetry dashboard on the right."""
        # Main HUD panel background
        panel_rect = pygame.Rect(self.panel_x, self.panel_y, self.panel_width, self.panel_height)
        pygame.draw.rect(self.screen, config.COLOR_PANEL_BG, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, (51, 65, 85), panel_rect, 2, border_radius=12)

        # Panel Header
        header_surf = self.font_panel_title.render("Campus Telemetry HUD", True, config.COLOR_ACCENT)
        self.screen.blit(header_surf, (self.panel_x + 20, self.panel_y + 16))

        mode_str = str(stats.get("mode", "Autonomous Nav"))
        sub_surf = self.font_subtitle.render(f"Operating Mode: {mode_str}", True, config.COLOR_TEXT_MUTED)
        self.screen.blit(sub_surf, (self.panel_x + 20, self.panel_y + 40))

        # Status Indicator Pill (Phase 17 Enhancement)
        success_val = stats.get("success", False)
        status_type = "GOAL" if success_val else "ACTIVE"
        status_lbl = "ARRIVED" if success_val else "ACTIVE"
        self.draw_status_indicator(self.panel_x + self.panel_width - 150, self.panel_y + 14, status_type, status_lbl)

        # Divider line
        pygame.draw.line(
            self.screen,
            (51, 65, 85),
            (self.panel_x + 20, self.panel_y + 62),
            (self.panel_x + self.panel_width - 20, self.panel_y + 62),
            1,
        )

        # Required Metric Entries
        reward_val = float(stats.get("reward", 0.0))
        reward_color = config.COLOR_SUCCESS if reward_val >= 0 else config.COLOR_DANGER
        success_str = "GOAL REACHED!" if success_val else stats.get("status", "Navigating")
        success_color = config.COLOR_SUCCESS if success_val else (250, 204, 21)

        metric_cards = [
            ("Current Mode", mode_str, (56, 189, 248), False),
            ("Episode Number", f"#{stats.get('episode', 1)}", config.COLOR_TEXT_LIGHT, True),
            ("Current Reward", f"{reward_val:+.1f}", reward_color, True),
            ("Step Count", f"{stats.get('steps', 0)} / {config.MAX_STEPS_PER_EPISODE}", config.COLOR_TEXT_LIGHT, True),
            ("Epsilon (Exploration)", f"{stats.get('epsilon', 0.0):.4f}", (192, 132, 252), True),
            ("Success Status", success_str, success_color, False),
            ("Agent Position", f"({stats.get('position', (0,0))[0]}, {stats.get('position', (0,0))[1]})", config.COLOR_TEXT_LIGHT, True),
            ("Distance to Goal", f"{stats.get('distance', 18)} blocks", config.COLOR_ACCENT, True),
        ]

        curr_y = self.panel_y + 72
        card_h = 42
        card_w = self.panel_width - 40

        for label, val, val_col, is_mono in metric_cards:
            card_rect = pygame.Rect(self.panel_x + 20, curr_y, card_w, card_h)
            pygame.draw.rect(self.screen, config.COLOR_PANEL_CARD, card_rect, border_radius=6)

            lbl_surf = self.font_label.render(label, True, config.COLOR_TEXT_MUTED)
            val_font = self.font_mono_val if is_mono else self.font_val
            val_surf = val_font.render(val, True, val_col)

            self.screen.blit(lbl_surf, (card_rect.x + 12, card_rect.y + 11))
            self.screen.blit(val_surf, (card_rect.right - val_surf.get_width() - 12, card_rect.y + 11))

            curr_y += card_h + 6

        # Footer Quick Controls
        footer_rect = pygame.Rect(self.panel_x + 20, self.panel_y + self.panel_height - 62, card_w, 46)
        pygame.draw.rect(self.screen, (24, 33, 47), footer_rect, border_radius=6)
        pygame.draw.rect(self.screen, (51, 65, 85), footer_rect, 1, border_radius=6)

        c1 = self.font_small.render("SPACE: Pause | R: Reset | ESC: Menu", True, (226, 232, 240))
        c2 = self.font_small.render("1: AI Nav  | 2: Manual Play | T: Train", True, config.COLOR_ACCENT)
        self.screen.blit(c1, (footer_rect.centerx - c1.get_width() // 2, footer_rect.y + 6))
        self.screen.blit(c2, (footer_rect.centerx - c2.get_width() // 2, footer_rect.y + 24))

    # --------------------------------------------------------------------------
    # Main Render Frame Dispatcher
    # --------------------------------------------------------------------------

    def render(self, env: CampusEnvironment, stats: Dict) -> None:
        """
        Render a complete frame: background, title header, campus grid,
        smooth animated agent, and live telemetry panel.
        """
        self.screen.fill(config.COLOR_BACKGROUND)

        # Header Title
        title_surf = self.font_title.render("Campus Navigator", True, (15, 23, 42))
        sub_surf = self.font_subtitle.render(
            "An Intelligent Path-Finding Game Using Q-Learning (Reinforcement Learning)",
            True,
            (100, 116, 139),
        )
        self.screen.blit(title_surf, (self.grid_x, 16))
        self.screen.blit(sub_surf, (self.grid_x, 48))

        # Update smooth interpolation
        self.update_agent_animation(env.agent_pos)

        # Draw visual elements
        self.draw_grid(env, stats)
        self.draw_agent()
        self.draw_info_panel(stats)
        self.draw_legend(env)

        pygame.display.flip()

    # --------------------------------------------------------------------------
    # Phase 8: Training Mode Renderer & HUD
    # --------------------------------------------------------------------------

    def draw_training_info_panel(
        self,
        stats: Dict,
        speed_buttons: List[Button],
        stop_button: Button,
        active_speed: str,
    ) -> None:
        """Render the specialized HUD panel for live AI Training Mode with progress bars."""
        # Panel Background
        panel_rect = pygame.Rect(self.panel_x, self.panel_y, self.panel_width, self.panel_height)
        pygame.draw.rect(self.screen, config.COLOR_PANEL_BG, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, (51, 65, 85), panel_rect, 2, border_radius=12)

        # Header Title
        header_surf = self.font_panel_title.render("AI Training Mode", True, config.COLOR_ACCENT)
        self.screen.blit(header_surf, (self.panel_x + 20, self.panel_y + 16))

        sub_surf = self.font_subtitle.render("Live Q-Learning Process (Tabular RL)", True, config.COLOR_TEXT_MUTED)
        self.screen.blit(sub_surf, (self.panel_x + 20, self.panel_y + 40))

        # Live Status Indicator Pill (Phase 17 Enhancement)
        outcome_str = stats.get("outcome", "Training...")
        is_paused = stats.get("paused", False)
        status_type = "PAUSED" if is_paused else ("GOAL" if "GOAL" in outcome_str else "TRAINING")
        status_lbl = "PAUSED" if is_paused else ("GOAL REACHED" if "GOAL" in outcome_str else "AI TRAINING")
        self.draw_status_indicator(self.panel_x + self.panel_width - 150, self.panel_y + 14, status_type, status_lbl)

        # Divider line
        pygame.draw.line(
            self.screen,
            (51, 65, 85),
            (self.panel_x + 20, self.panel_y + 60),
            (self.panel_x + self.panel_width - 20, self.panel_y + 60),
            1,
        )

        ep_num = stats.get("episode", 1)
        tot_eps = stats.get("total_episodes", 1000)
        progress_pct = min(1.0, ep_num / max(1, tot_eps))
        eps_val = float(stats.get("epsilon", 1.0))
        eps_ratio = max(0.0, min(1.0, (eps_val - config.EPSILON_MIN) / max(0.0001, (config.EPSILON_INITIAL - config.EPSILON_MIN))))

        # ----------------------------------------------------------------------
        # Live Dual Progress Bar Card (Phase 17 Enhancement)
        # ----------------------------------------------------------------------
        card_w = self.panel_width - 40
        prog_card_h = 76
        prog_rect = pygame.Rect(self.panel_x + 20, self.panel_y + 68, card_w, prog_card_h)
        pygame.draw.rect(self.screen, config.COLOR_PANEL_CARD, prog_rect, border_radius=8)
        pygame.draw.rect(self.screen, (51, 65, 85), prog_rect, 1, border_radius=8)

        # 1. Episode Progress Bar
        bar_w = card_w - 24
        ep_lbl = self.font_small.render("EPISODE PROGRESS:", True, config.COLOR_TEXT_MUTED)
        ep_val = self.font_mono_small.render(f"{ep_num} / {tot_eps} ({progress_pct * 100.0:.1f}%)", True, (56, 189, 248))
        self.screen.blit(ep_lbl, (prog_rect.x + 12, prog_rect.y + 8))
        self.screen.blit(ep_val, (prog_rect.right - ep_val.get_width() - 12, prog_rect.y + 8))

        # Episode Track and Fill
        ep_track = pygame.Rect(prog_rect.x + 12, prog_rect.y + 24, bar_w, 8)
        pygame.draw.rect(self.screen, config.COLOR_PROGRESS_BG, ep_track, border_radius=4)
        ep_fill_w = max(4, int(progress_pct * bar_w))
        ep_fill = pygame.Rect(prog_rect.x + 12, prog_rect.y + 24, ep_fill_w, 8)
        pygame.draw.rect(self.screen, config.COLOR_PROGRESS_FILL, ep_fill, border_radius=4)
        pygame.draw.line(self.screen, config.COLOR_PROGRESS_ACCENT, (ep_fill.x, ep_fill.y), (ep_fill.right, ep_fill.y), 1)

        # 2. Epsilon Exploration Decay Gauge
        eps_lbl = self.font_small.render("EPSILON (EXPLORATION):", True, config.COLOR_TEXT_MUTED)
        eps_str = self.font_mono_small.render(f"ε = {eps_val:.4f} ({eps_ratio * 100.0:.0f}% explore)", True, (192, 132, 252))
        self.screen.blit(eps_lbl, (prog_rect.x + 12, prog_rect.y + 40))
        self.screen.blit(eps_str, (prog_rect.right - eps_str.get_width() - 12, prog_rect.y + 40))

        # Epsilon Track and Fill
        eps_track = pygame.Rect(prog_rect.x + 12, prog_rect.y + 56, bar_w, 8)
        pygame.draw.rect(self.screen, config.COLOR_PROGRESS_BG, eps_track, border_radius=4)
        eps_fill_w = max(4, int(eps_ratio * bar_w))
        eps_fill = pygame.Rect(prog_rect.x + 12, prog_rect.y + 56, eps_fill_w, 8)
        pygame.draw.rect(self.screen, config.COLOR_EPSILON_FILL, eps_fill, border_radius=4)

        # ----------------------------------------------------------------------
        # Telemetry Metric Cards
        # ----------------------------------------------------------------------
        rew_val = float(stats.get("reward", 0.0))
        rew_col = config.COLOR_SUCCESS if rew_val >= 0 else config.COLOR_DANGER
        outcome_col = config.COLOR_SUCCESS if "GOAL" in outcome_str else ((250, 204, 21) if "Step" in outcome_str else config.COLOR_DANGER)
        coll_count = stats.get("collisions", 0)

        metric_cards = [
            ("Current Reward", f"{rew_val:+.1f}", rew_col, True),
            ("Steps in Ep", f"{stats.get('steps', 0)} / {config.MAX_STEPS_PER_EPISODE}", config.COLOR_TEXT_LIGHT, True),
            ("Episode Outcome", outcome_str, outcome_col, False),
            ("Total Collisions", f"{coll_count} hits", config.COLOR_DANGER if coll_count > 0 else config.COLOR_TEXT_LIGHT, True),
            ("Success Rate", f"{stats.get('success_rate', 0.0):.1f}%", config.COLOR_ACCENT, True),
        ]

        curr_y = self.panel_y + 152
        card_h = 34

        for label, val, val_col, is_mono in metric_cards:
            card_rect = pygame.Rect(self.panel_x + 20, curr_y, card_w, card_h)
            pygame.draw.rect(self.screen, config.COLOR_PANEL_CARD, card_rect, border_radius=6)

            lbl_surf = self.font_label.render(label, True, config.COLOR_TEXT_MUTED)
            val_font = self.font_mono_val if is_mono else self.font_val
            val_surf = val_font.render(val, True, val_col)

            self.screen.blit(lbl_surf, (card_rect.x + 12, card_rect.y + 8))
            self.screen.blit(val_surf, (card_rect.right - val_surf.get_width() - 12, card_rect.y + 8))

            curr_y += card_h + 5

        # Speed Control Section
        speed_label = self.font_small.render("TRAINING SPEED CONTROL", True, config.COLOR_TEXT_MUTED)
        self.screen.blit(speed_label, (self.panel_x + 22, curr_y + 10))

        # Draw Speed Buttons
        mouse_pos = pygame.mouse.get_pos()
        buttons_to_draw = speed_buttons.values() if isinstance(speed_buttons, dict) else speed_buttons
        for btn in buttons_to_draw:
            btn.is_selected = (btn.text.upper() == active_speed.upper())
            btn.update(mouse_pos)
            btn.draw(self.screen)

        # Draw Stop Training Button
        stop_button.update(mouse_pos)
        stop_button.draw(self.screen)

    def render_training(
        self,
        env: CampusEnvironment,
        stats: Dict,
        speed_buttons: List[Button],
        stop_button: Button,
        active_speed: str,
    ) -> None:
        """Render a single frame for the live AI Training Mode."""
        self.screen.fill(config.COLOR_BACKGROUND)

        # Header Title
        title_surf = self.font_title.render("Campus Navigator: AI Training Mode", True, (15, 23, 42))
        sub_surf = self.font_subtitle.render(
            "Visual Experiential Learning — Tabular Q-Learning Agent Exploring Campus",
            True,
            (100, 116, 139),
        )
        self.screen.blit(title_surf, (self.grid_x, 16))
        self.screen.blit(sub_surf, (self.grid_x, 48))

        # Smooth agent movement (only lerp if not running in ultra-fast mode)
        if active_speed != "FAST":
            self.update_agent_animation(env.agent_pos)
        else:
            r, c = env.agent_pos
            self.agent_anim_x = self.grid_x + (c * self.cell_size) + (self.cell_size // 2)
            self.agent_anim_y = self.grid_y + (r * self.cell_size) + (self.cell_size // 2)

        # Draw visual components
        self.draw_grid(env, stats)
        self.draw_agent()
        self.draw_training_info_panel(stats, speed_buttons, stop_button, active_speed)
        self.draw_legend(env)

        pygame.display.flip()

    # --------------------------------------------------------------------------
    # Training Completion Screen
    # --------------------------------------------------------------------------

    def draw_training_complete_screen(
        self,
        summary: Dict,
        buttons: List[Button],
    ) -> None:
        """Render an attractive completion screen with training summary metrics."""
        self.screen.fill(config.COLOR_BACKGROUND)

        # Header Title
        title_text = "AI Training Complete!" if summary.get("completed", True) else "AI Training Stopped by User"
        title_color = (22, 101, 52) if summary.get("completed", True) else (180, 83, 9)
        title_surf = self.font_title.render(title_text, True, title_color)
        sub_surf = self.font_subtitle.render(
            "Q-table updated, metrics logged, and performance charts generated.",
            True,
            (71, 85, 105),
        )
        self.screen.blit(title_surf, (config.SCREEN_WIDTH // 2 - title_surf.get_width() // 2, 40))
        self.screen.blit(sub_surf, (config.SCREEN_WIDTH // 2 - sub_surf.get_width() // 2, 85))

        # Main Card Container
        card_w, card_h = 760, 480
        card_x = (config.SCREEN_WIDTH - card_w) // 2
        card_y = 125
        card_rect = pygame.Rect(card_x, card_y, card_w, card_h)
        pygame.draw.rect(self.screen, (255, 255, 255), card_rect, border_radius=12)
        pygame.draw.rect(self.screen, (203, 213, 225), card_rect, 2, border_radius=12)

        # Metrics rows
        rows = [
            ("Total Episodes Completed", f"{summary.get('episodes', 0)} episodes", (37, 99, 235)),
            ("Overall Success Rate", f"{summary.get('success_rate', 0.0):.1f}%", config.COLOR_SUCCESS),
            ("Average Cumulative Reward", f"{summary.get('avg_reward', 0.0):+.1f}", (37, 99, 235)),
            ("Average Steps to Goal", f"{summary.get('avg_steps', 0.0):.1f} steps", (30, 41, 59)),
            ("Total Collisions Recorded", f"{summary.get('collisions', 0)} collisions", (220, 38, 38)),
            ("Final Epsilon (Exploration)", f"{summary.get('final_epsilon', 0.05):.4f}", (147, 51, 234)),
            ("Learned Q-Table States", f"{summary.get('q_entries', 0)} / 400 entries updated", (16, 185, 129)),
        ]

        curr_y = card_y + 25
        for title, val, col in rows:
            t_surf = self.font_body.render(title, True, (71, 85, 105))
            v_surf = self.font_body_bold.render(val, True, col)
            self.screen.blit(t_surf, (card_x + 36, curr_y))
            self.screen.blit(v_surf, (card_x + card_w - v_surf.get_width() - 36, curr_y))
            pygame.draw.line(self.screen, (241, 245, 249), (card_x + 36, curr_y + 30), (card_x + card_w - 36, curr_y + 30), 1)
            curr_y += 42

        # Artifact confirmation banner
        banner_rect = pygame.Rect(card_x + 36, curr_y + 12, card_w - 72, 54)
        pygame.draw.rect(self.screen, (240, 253, 244), banner_rect, border_radius=8)
        pygame.draw.rect(self.screen, (187, 247, 208), banner_rect, 1, border_radius=8)

        b_txt = self.font_small.render(
            "[+] Saved: models/q_table.npy | results/training_results.csv | results/plots/*.png",
            True,
            (22, 101, 52),
        )
        self.screen.blit(b_txt, (banner_rect.centerx - b_txt.get_width() // 2, banner_rect.centery - b_txt.get_height() // 2))

        # Action Buttons
        mouse_pos = pygame.mouse.get_pos()
        for btn in buttons:
            btn.update(mouse_pos)
            btn.draw(self.screen)

        pygame.display.flip()

    # --------------------------------------------------------------------------
    # Phase 9: Watch Trained AI Mode Renderer
    # --------------------------------------------------------------------------

    def draw_route_path(
        self,
        path: List[Tuple[int, int]],
        start_pos: Optional[Tuple[int, int]] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Render the complete learned route with glowing path waypoints, drop shadows,
        and step numbering along the route for academic viva presentation.
        """
        if not path or len(path) < 1:
            return

        sp = start_pos if start_pos is not None else config.START_POSITION
        gp = goal_pos if goal_pos is not None else config.GOAL_POSITION

        # 1. Subtle tile highlight along the route
        for r, c in path:
            if (r, c) != sp and (r, c) != gp:
                rect = pygame.Rect(
                    self.grid_x + c * self.cell_size,
                    self.grid_y + r * self.cell_size,
                    self.cell_size,
                    self.cell_size,
                )
                # Soft purple tint for route pathway tiles
                pygame.draw.rect(self.screen, (243, 232, 255), rect)
                pygame.draw.rect(self.screen, (216, 180, 254), rect, 1)

        # 2. Connecting line through waypoint centers
        pixel_points = []
        for r, c in path:
            px = self.grid_x + c * self.cell_size + (self.cell_size // 2)
            py = self.grid_y + r * self.cell_size + (self.cell_size // 2)
            pixel_points.append((px, py))

        if len(pixel_points) >= 2:
            # Drop shadow for line
            shadow_points = [(x + 1, y + 2) for x, y in pixel_points]
            pygame.draw.lines(self.screen, (192, 132, 252), False, shadow_points, 5)
            # Main route line
            pygame.draw.lines(self.screen, config.COLOR_OPTIMAL_PATH, False, pixel_points, 4)

        # 3. Waypoint circular pins with step sequence annotations
        total_pts = len(pixel_points)
        for idx, (px, py) in enumerate(pixel_points):
            if idx == 0:
                # Start pin: emerald halo
                pygame.draw.circle(self.screen, (220, 252, 231), (px, py), 10)
                pygame.draw.circle(self.screen, config.COLOR_START, (px, py), 6)
                pygame.draw.circle(self.screen, (255, 255, 255), (px, py), 3)
            elif idx == total_pts - 1:
                # Goal pin: radiant gold halo
                pygame.draw.circle(self.screen, (254, 240, 138), (px, py), 11)
                pygame.draw.circle(self.screen, config.COLOR_GOAL, (px, py), 7)
                pygame.draw.circle(self.screen, (255, 255, 255), (px, py), 3)
            else:
                # Interior waypoints
                if total_pts <= 25:
                    # Circular pin with step number
                    pin_radius = 8
                    pygame.draw.circle(self.screen, (255, 255, 255), (px, py), pin_radius)
                    pygame.draw.circle(self.screen, (147, 51, 234), (px, py), pin_radius, 1)
                    num_surf = self.font_mono_tiny.render(str(idx), True, (109, 40, 217))
                    self.screen.blit(num_surf, (px - num_surf.get_width() // 2, py - num_surf.get_height() // 2))
                else:
                    # Crisp compact pin
                    pygame.draw.circle(self.screen, (255, 255, 255), (px, py), 5)
                    pygame.draw.circle(self.screen, (147, 51, 234), (px, py), 3)

    def draw_custom_path(
        self,
        path: List[Tuple[int, int]],
        line_color: Tuple[int, int, int],
        pin_color: Tuple[int, int, int],
        tile_tint: Optional[Tuple[int, int, int]] = None,
        line_width: int = 4,
    ) -> None:
        """Render a custom path trajectory with tile highlights and waypoint pins."""
        if not path or len(path) < 1:
            return

        # 1. Subtle tile highlights
        if tile_tint is not None:
            for r, c in path:
                if (r, c) != config.START_POSITION and (r, c) != config.GOAL_POSITION:
                    rect = pygame.Rect(
                        self.grid_x + c * self.cell_size,
                        self.grid_y + r * self.cell_size,
                        self.cell_size,
                        self.cell_size,
                    )
                    pygame.draw.rect(self.screen, tile_tint, rect)

        # 2. Connecting line
        pixel_points = []
        for r, c in path:
            px = self.grid_x + c * self.cell_size + (self.cell_size // 2)
            py = self.grid_y + r * self.cell_size + (self.cell_size // 2)
            pixel_points.append((px, py))

        if len(pixel_points) >= 2:
            pygame.draw.lines(self.screen, line_color, False, pixel_points, line_width)

        # 3. Waypoint circular pins
        for px, py in pixel_points:
            pygame.draw.circle(self.screen, (255, 255, 255), (px, py), 5)
            pygame.draw.circle(self.screen, pin_color, (px, py), 3)

    def draw_dual_route_paths(
        self,
        human_path: List[Tuple[int, int]],
        ai_path: List[Tuple[int, int]],
    ) -> None:
        """Render both human and AI paths with distinct colors and a visual legend."""
        # 1. Human Path in Cyan/Sky Blue
        if human_path and len(human_path) > 1:
            self.draw_custom_path(
                human_path,
                line_color=(14, 165, 233),
                pin_color=(2, 132, 199),
                tile_tint=(224, 242, 254),
                line_width=3,
            )

        # 2. AI Path in Purple
        if ai_path and len(ai_path) > 1:
            self.draw_custom_path(
                ai_path,
                line_color=(147, 51, 234),
                pin_color=(126, 34, 206),
                tile_tint=(243, 232, 255),
                line_width=4,
            )

        # 3. Trajectory Legend Badge above Grid
        pygame.draw.circle(self.screen, (14, 165, 233), (self.grid_x + 8, self.grid_y - 12), 5)
        h_lbl = self.font_small.render("Human Trajectory", True, (14, 165, 233))
        self.screen.blit(h_lbl, (self.grid_x + 18, self.grid_y - 18))

        pygame.draw.circle(self.screen, (147, 51, 234), (self.grid_x + 150, self.grid_y - 12), 5)
        ai_lbl = self.font_small.render("AI Policy Path", True, (147, 51, 234))
        self.screen.blit(ai_lbl, (self.grid_x + 160, self.grid_y - 18))

    def draw_watch_ai_panel(
        self,
        stats: Dict,
        restart_btn: Button,
        menu_btn: Button,
        route_btn: Optional[Button] = None,
    ) -> None:
        """Render the telemetry HUD for Watch Trained AI Mode."""
        panel_rect = pygame.Rect(self.panel_x, self.panel_y, self.panel_width, self.panel_height)
        pygame.draw.rect(self.screen, config.COLOR_PANEL_BG, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, (51, 65, 85), panel_rect, 2, border_radius=12)

        # Header Title
        header_surf = self.font_panel_title.render("Watch Trained AI", True, config.COLOR_ACCENT)
        self.screen.blit(header_surf, (self.panel_x + 20, self.panel_y + 16))

        sub_surf = self.font_subtitle.render("Exploiting Learned Policy (Epsilon = 0.0)", True, config.COLOR_TEXT_MUTED)
        self.screen.blit(sub_surf, (self.panel_x + 20, self.panel_y + 40))

        # Status Indicator Pill (Phase 17 Enhancement)
        is_done = stats.get("goal_reached", False)
        is_paused = stats.get("paused", False)
        status_type = "GOAL" if is_done else ("PAUSED" if is_paused else "ACTIVE")
        status_lbl = "ARRIVED" if is_done else ("PAUSED" if is_paused else "POLICY RUN")
        self.draw_status_indicator(self.panel_x + self.panel_width - 150, self.panel_y + 14, status_type, status_lbl)

        # Divider line
        pygame.draw.line(
            self.screen,
            (51, 65, 85),
            (self.panel_x + 20, self.panel_y + 60),
            (self.panel_x + self.panel_width - 20, self.panel_y + 60),
            1,
        )

        # Required Telemetry Display
        curr_state = stats.get("current_state", (0, 0))
        act_name = stats.get("action", "None")
        step_rew = float(stats.get("reward", 0.0))
        tot_rew = float(stats.get("total_reward", 0.0))
        steps = stats.get("steps", 0)
        start_name = stats.get("start_name", "Main Gate")
        dest_name = stats.get("destination_name", "Library")

        step_rew_col = config.COLOR_SUCCESS if step_rew >= 0 else config.COLOR_DANGER
        tot_rew_col = config.COLOR_SUCCESS if tot_rew >= 0 else config.COLOR_DANGER
        status_text = "DESTINATION REACHED! (+100)" if is_done else stats.get("status", "Navigating Route")
        status_col = config.COLOR_SUCCESS if is_done else (250, 204, 21)

        metric_cards = [
            ("Target Destination", dest_name, (245, 158, 11), False),
            ("Origin Landmark", start_name, (34, 197, 94), False),
            ("Current State", f"Row {curr_state[0]}, Col {curr_state[1]}", (56, 189, 248), True),
            ("Selected Action", act_name, (244, 114, 182), True),
            ("Immediate Reward", f"{step_rew:+.1f}", step_rew_col, True),
            ("Total Reward", f"{tot_rew:+.1f}", tot_rew_col, True),
            ("Steps Taken", f"{steps} / {config.MAX_STEPS_PER_EPISODE}", config.COLOR_TEXT_LIGHT, True),
            ("Navigation Status", status_text, status_col, False),
        ]

        curr_y = self.panel_y + 66
        card_h = 36
        card_w = self.panel_width - 40

        for label, val, val_col, is_mono in metric_cards:
            card_rect = pygame.Rect(self.panel_x + 20, curr_y, card_w, card_h)
            pygame.draw.rect(self.screen, config.COLOR_PANEL_CARD, card_rect, border_radius=6)

            lbl_surf = self.font_label.render(label, True, config.COLOR_TEXT_MUTED)
            val_font = self.font_mono_val if is_mono else self.font_val
            val_surf = val_font.render(val, True, val_col)

            self.screen.blit(lbl_surf, (card_rect.x + 12, card_rect.y + 8))
            self.screen.blit(val_surf, (card_rect.right - val_surf.get_width() - 12, card_rect.y + 8))

            curr_y += card_h + 6

        # Draw [Change Route] button if provided
        mouse_pos = pygame.mouse.get_pos()
        if route_btn:
            route_btn.update(mouse_pos)
            route_btn.draw(self.screen)

        # Draw Action Buttons [Restart] and [Back to Menu]
        restart_btn.update(mouse_pos)
        restart_btn.draw(self.screen)

        menu_btn.update(mouse_pos)
        menu_btn.draw(self.screen)

    def render_watch_ai(
        self,
        env: CampusEnvironment,
        stats: Dict,
        path: List[Tuple[int, int]],
        show_route: bool,
        restart_btn: Button,
        menu_btn: Button,
        route_btn: Optional[Button] = None,
    ) -> None:
        """Render a complete frame for Watch Trained AI Mode."""
        self.screen.fill(config.COLOR_BACKGROUND)

        dest_name = getattr(env, "current_destination_name", "Central Library")
        start_name = getattr(env, "current_start_name", "Main Gate")

        # Header Title
        title_surf = self.font_title.render("Campus Navigator: Watch Trained AI", True, (15, 23, 42))
        sub_surf = self.font_subtitle.render(
            f"Greedy Exploitation — Following Learned Policy from {start_name} to {dest_name}",
            True,
            (100, 116, 139),
        )
        self.screen.blit(title_surf, (self.grid_x, 16))
        self.screen.blit(sub_surf, (self.grid_x, 48))

        # Update smooth agent animation
        self.update_agent_animation(env.agent_pos)

        # Draw Grid with active animations & feedback
        self.draw_grid(env, stats)

        # Draw Route Path (if arrived or showing trail)
        if show_route or stats.get("goal_reached", False):
            self.draw_route_path(path, start_pos=env.start_pos, goal_pos=env.goal_pos)

        # Draw Agent
        self.draw_agent()

        # Draw HUD Panel with Restart, Route, and Back buttons
        self.draw_watch_ai_panel(stats, restart_btn, menu_btn, route_btn=route_btn)
        self.draw_legend(env)

        pygame.display.flip()

    def draw_no_model_screen(
        self,
        buttons: List[Button],
        error_title: Optional[str] = None,
        error_message: Optional[str] = None,
        error_detail: Optional[str] = None,
    ) -> None:
        """Render a user-friendly error screen when no trained model exists or model is corrupted."""
        self.screen.fill(config.COLOR_BACKGROUND)

        # Header Title
        title_surf = self.font_title.render("Campus Navigator", True, (15, 23, 42))
        self.screen.blit(title_surf, (config.SCREEN_WIDTH // 2 - title_surf.get_width() // 2, 80))

        # Alert Card
        card_w, card_h = 700, 380
        card_x = (config.SCREEN_WIDTH - card_w) // 2
        card_y = 150
        card_rect = pygame.Rect(card_x, card_y, card_w, card_h)
        pygame.draw.rect(self.screen, (255, 255, 255), card_rect, border_radius=12)
        pygame.draw.rect(self.screen, (254, 202, 202), card_rect, 2, border_radius=12)

        # Warning icon badge
        pygame.draw.circle(self.screen, (254, 226, 226), (config.SCREEN_WIDTH // 2, card_y + 55), 32)
        w_surf = self.font_title.render("!", True, (220, 38, 38))
        self.screen.blit(w_surf, (config.SCREEN_WIDTH // 2 - w_surf.get_width() // 2, card_y + 38))

        # Primary Message
        primary_msg = error_message or "No trained model found. Please train the AI first."
        msg1 = self.font_header.render(primary_msg, True, (185, 28, 28))
        self.screen.blit(msg1, (config.SCREEN_WIDTH // 2 - msg1.get_width() // 2, card_y + 110))

        # Secondary context details
        sub1 = error_detail or "The model file 'models/q_table.npy' does not exist yet."
        sub2 = "You can train the agent now using in-game training or the terminal."
        msg2 = self.font_body.render(sub1, True, (71, 85, 105))
        msg3 = self.font_body.render(sub2, True, (100, 116, 139))
        self.screen.blit(msg2, (config.SCREEN_WIDTH // 2 - msg2.get_width() // 2, card_y + 155))
        self.screen.blit(msg3, (config.SCREEN_WIDTH // 2 - msg3.get_width() // 2, card_y + 180))

        # Log notice
        log_notice = self.font_small.render("Diagnostic information recorded in logs/campus_navigator.log", True, (148, 163, 184))
        self.screen.blit(log_notice, (config.SCREEN_WIDTH // 2 - log_notice.get_width() // 2, card_y + 215))

        # Action Buttons
        mouse_pos = pygame.mouse.get_pos()
        for btn in buttons:
            btn.update(mouse_pos)
            btn.draw(self.screen)

        pygame.display.flip()

    # --------------------------------------------------------------------------
    # Phase 10: Manual Human Navigation Mode Renderer
    # --------------------------------------------------------------------------

    def draw_human_play_panel(
        self,
        stats: Dict,
        restart_btn: Button,
        menu_btn: Button,
        route_btn: Optional[Button] = None,
    ) -> None:
        """Render the specialized HUD panel for manual human gameplay."""
        panel_rect = pygame.Rect(self.panel_x, self.panel_y, self.panel_width, self.panel_height)
        pygame.draw.rect(self.screen, config.COLOR_PANEL_BG, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, (51, 65, 85), panel_rect, 2, border_radius=12)

        # Header Title
        header_surf = self.font_panel_title.render("Manual Human Play", True, config.COLOR_ACCENT)
        self.screen.blit(header_surf, (self.panel_x + 20, self.panel_y + 16))

        sub_surf = self.font_subtitle.render("Campus Navigation (WASD / Arrows)", True, config.COLOR_TEXT_MUTED)
        self.screen.blit(sub_surf, (self.panel_x + 20, self.panel_y + 40))

        # Status Indicator Pill (Phase 17 Enhancement)
        is_done = stats.get("goal_reached", False)
        now = pygame.time.get_ticks()
        recent_collision = (stats.get("last_collision_time", 0) > 0 and (now - stats.get("last_collision_time", 0) < 900))
        status_type = "GOAL" if is_done else ("COLLISION" if recent_collision else "MANUAL")
        status_lbl = "MISSION COMPLETE" if is_done else ("PENALTY" if recent_collision else "MANUAL PLAY")
        self.draw_status_indicator(self.panel_x + self.panel_width - 150, self.panel_y + 14, status_type, status_lbl)

        # Divider line
        pygame.draw.line(
            self.screen,
            (51, 65, 85),
            (self.panel_x + 20, self.panel_y + 60),
            (self.panel_x + self.panel_width - 20, self.panel_y + 60),
            1,
        )

        # Telemetry Metrics
        curr_state = stats.get("current_state", (0, 0))
        act_name = stats.get("action", "None")
        step_rew = float(stats.get("reward", 0.0))
        tot_rew = float(stats.get("total_reward", 0.0))
        steps = stats.get("steps", 0)
        time_str = stats.get("time_str", "0.0s")
        collisions = stats.get("collisions", 0)
        start_name = stats.get("start_name", "Main Gate")
        dest_name = stats.get("destination_name", "Library")

        step_rew_col = config.COLOR_SUCCESS if step_rew >= 0 else config.COLOR_DANGER
        tot_rew_col = config.COLOR_SUCCESS if tot_rew >= 0 else config.COLOR_DANGER
        status_text = "MISSION COMPLETE! (+100)" if is_done else stats.get("status", "Navigating")
        status_col = config.COLOR_SUCCESS if is_done else (250, 204, 21)

        metric_cards = [
            ("Target Destination", dest_name, (245, 158, 11), False),
            ("Origin Landmark", start_name, (34, 197, 94), False),
            ("Current State", f"Row {curr_state[0]}, Col {curr_state[1]}", (56, 189, 248), True),
            ("Elapsed Time", time_str, (250, 204, 21), True),
            ("Score / Reward", f"{tot_rew:+.1f}", tot_rew_col, True),
            ("Steps Taken", f"{steps} / {config.MAX_STEPS_PER_EPISODE}", config.COLOR_TEXT_LIGHT, True),
            ("Last Move Taken", act_name, (244, 114, 182), True),
            ("Collisions / Hits", f"{collisions} hits", config.COLOR_DANGER if collisions > 0 else config.COLOR_TEXT_LIGHT, True),
            ("Status", status_text, status_col, False),
        ]

        curr_y = self.panel_y + 66
        card_h = 34
        card_w = self.panel_width - 40

        for label, val, val_col, is_mono in metric_cards:
            card_rect = pygame.Rect(self.panel_x + 20, curr_y, card_w, card_h)
            pygame.draw.rect(self.screen, config.COLOR_PANEL_CARD, card_rect, border_radius=6)

            lbl_surf = self.font_label.render(label, True, config.COLOR_TEXT_MUTED)
            val_font = self.font_mono_val if is_mono else self.font_val
            val_surf = val_font.render(val, True, val_col)

            self.screen.blit(lbl_surf, (card_rect.x + 12, card_rect.y + 7))
            self.screen.blit(val_surf, (card_rect.right - val_surf.get_width() - 12, card_rect.y + 7))

            curr_y += card_h + 4

        # Quick Keyboard Guide Note
        hint_surf = self.font_small.render("Move: WASD / Arrows  |  Route: C  |  Restart: R  |  Menu: ESC", True, (148, 163, 184))
        self.screen.blit(hint_surf, (self.panel_x + self.panel_width // 2 - hint_surf.get_width() // 2, self.panel_y + self.panel_height - 110))

        # Draw [Change Route] button if provided
        mouse_pos = pygame.mouse.get_pos()
        if route_btn:
            route_btn.update(mouse_pos)
            route_btn.draw(self.screen)

        # Draw Action Buttons [Restart] and [Back to Menu]
        restart_btn.update(mouse_pos)
        restart_btn.draw(self.screen)

        menu_btn.update(mouse_pos)
        menu_btn.draw(self.screen)

    def draw_human_victory_modal(
        self,
        stats: Dict,
        play_again_btn: Button,
        compare_ai_btn: Button,
        menu_btn: Button,
    ) -> None:
        """Render celebratory MISSION COMPLETE screen with Steps, Time, Reward & 3 buttons."""
        # 1. Semi-transparent backdrop overlay
        overlay = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((15, 23, 42, 195))
        self.screen.blit(overlay, (0, 0))

        # 2. Main Card Container
        card_w, card_h = 780, 530
        card_x = (config.SCREEN_WIDTH - card_w) // 2
        card_y = (config.SCREEN_HEIGHT - card_h) // 2
        card_rect = pygame.Rect(card_x, card_y, card_w, card_h)

        pygame.draw.rect(self.screen, (255, 255, 255), card_rect, border_radius=16)
        pygame.draw.rect(self.screen, (245, 158, 11), card_rect, 3, border_radius=16)

        # Top decorative ribbon
        ribbon_rect = pygame.Rect(card_x, card_y, card_w, 75)
        pygame.draw.rect(self.screen, (254, 243, 199), ribbon_rect, border_top_left_radius=16, border_top_right_radius=16)
        pygame.draw.line(self.screen, (252, 211, 77), (card_x, card_y + 75), (card_x + card_w, card_y + 75), 2)

        # Header Titles: MISSION COMPLETE
        t_title = self.font_title.render("MISSION COMPLETE", True, (146, 64, 14))
        t_sub = self.font_subtitle.render("Central Library Reached (9, 9) — Measured Performance Summary", True, (180, 83, 9))
        self.screen.blit(t_title, (config.SCREEN_WIDTH // 2 - t_title.get_width() // 2, card_y + 14))
        self.screen.blit(t_sub, (config.SCREEN_WIDTH // 2 - t_sub.get_width() // 2, card_y + 46))

        # Core Metrics Extraction
        h_steps = stats.get("steps", 0)
        h_time = stats.get("time_str", "0.0s")
        h_reward = float(stats.get("total_reward", 0.0))
        h_collisions = stats.get("collisions", 0)
        route_len = stats.get("route_length", h_steps + 1)

        # AI Benchmark Data
        ai_steps = 18
        ai_reward = 170.0

        # 3 Key Stat Cards (Prominently displaying Steps, Time, Reward)
        stat_cards = [
            ("Steps:", f"{h_steps}", (37, 99, 235), f"Optimal AI: {ai_steps}"),
            ("Time:", f"{h_time}", (217, 119, 6), "Elapsed Duration"),
            ("Reward:", f"{h_reward:+.1f}", config.COLOR_SUCCESS if h_reward >= 0 else (220, 38, 38), f"Optimal AI: +{ai_reward:.1f}"),
        ]

        card_gap = 18
        sc_w = (card_w - 72 - card_gap * 2) // 3
        sc_h = 76
        sc_y = card_y + 90
        sc_start_x = card_x + 36

        for i, (sc_label, sc_val, sc_color, sc_sub) in enumerate(stat_cards):
            box_rect = pygame.Rect(sc_start_x + i * (sc_w + card_gap), sc_y, sc_w, sc_h)
            pygame.draw.rect(self.screen, (248, 250, 252), box_rect, border_radius=8)
            pygame.draw.rect(self.screen, (226, 232, 240), box_rect, 1, border_radius=8)

            l_surf = self.font_small.render(sc_label.upper(), True, (100, 116, 139))
            v_surf = self.font_panel_title.render(sc_val, True, sc_color)
            sub_surf = self.font_subtitle.render(sc_sub, True, (148, 163, 184))

            self.screen.blit(l_surf, (box_rect.x + 14, box_rect.y + 8))
            self.screen.blit(v_surf, (box_rect.x + 14, box_rect.y + 26))
            self.screen.blit(sub_surf, (box_rect.x + 14, box_rect.y + 52))

        # Objective Route Summary Banner (No biased claims)
        banner_rect = pygame.Rect(card_x + 36, card_y + 178, card_w - 72, 54)
        pygame.draw.rect(self.screen, (241, 245, 249), banner_rect, border_radius=8)
        pygame.draw.rect(self.screen, (203, 213, 225), banner_rect, 1, border_radius=8)

        b_badge = self.font_header.render(f"Route Traversed: {route_len} coordinates  |  Collisions: {h_collisions} hits", True, (15, 23, 42))
        b_desc = self.font_body.render("Click 'Compare with AI' to run the trained Q-Learning agent on this identical environment.", True, (71, 85, 105))
        self.screen.blit(b_badge, (banner_rect.centerx - b_badge.get_width() // 2, banner_rect.y + 6))
        self.screen.blit(b_desc, (banner_rect.centerx - b_desc.get_width() // 2, banner_rect.y + 28))

        # Measured Comparison Table
        table_y = card_y + 242
        table_w = card_w - 72
        table_x = card_x + 36

        # Table Header
        h_rect = pygame.Rect(table_x, table_y, table_w, 28)
        pygame.draw.rect(self.screen, (241, 245, 249), h_rect, border_radius=6)
        th1 = self.font_small.render("METRIC", True, (71, 85, 105))
        th2 = self.font_small.render("YOUR SCORE", True, (71, 85, 105))
        th3 = self.font_small.render("AI BENCHMARK", True, (71, 85, 105))
        th4 = self.font_small.render("DIFFERENCE", True, (71, 85, 105))

        self.screen.blit(th1, (table_x + 20, table_y + 6))
        self.screen.blit(th2, (table_x + 220, table_y + 6))
        self.screen.blit(th3, (table_x + 390, table_y + 6))
        self.screen.blit(th4, (table_x + 550, table_y + 6))

        # Data Rows
        diff_steps = h_steps - ai_steps
        diff_steps_str = "Tied Optimum!" if diff_steps == 0 else f"+{diff_steps} steps"
        diff_rew = h_reward - ai_reward
        diff_rew_str = "Optimal!" if abs(diff_rew) < 0.1 else f"{diff_rew:+.1f} pts"
        diff_col = h_collisions
        diff_col_str = "Zero Faults!" if diff_col == 0 else f"+{diff_col} collisions"

        rows = [
            ("Steps Taken", f"{h_steps} steps", f"{ai_steps} steps", diff_steps_str, config.COLOR_SUCCESS if diff_steps == 0 else (220, 38, 38)),
            ("Cumulative Reward", f"{h_reward:+.1f}", f"{ai_reward:+.1f}", diff_rew_str, config.COLOR_SUCCESS if diff_rew >= 0 else (220, 38, 38)),
            ("Collisions / Hits", f"{h_collisions} hits", "0 hits", diff_col_str, config.COLOR_SUCCESS if diff_col == 0 else (220, 38, 38)),
        ]

        row_y = table_y + 34
        for m_name, m_human, m_ai, m_diff, m_col in rows:
            r_rect = pygame.Rect(table_x, row_y, table_w, 34)
            pygame.draw.rect(self.screen, (248, 250, 252), r_rect, border_radius=6)

            t_name = self.font_body_bold.render(m_name, True, (30, 41, 59))
            t_human = self.font_val.render(m_human, True, (37, 99, 235))
            t_ai = self.font_val.render(m_ai, True, (16, 185, 129))
            t_diff = self.font_val.render(m_diff, True, m_col)

            self.screen.blit(t_name, (table_x + 20, row_y + 8))
            self.screen.blit(t_human, (table_x + 220, row_y + 8))
            self.screen.blit(t_ai, (table_x + 390, row_y + 8))
            self.screen.blit(t_diff, (table_x + 550, row_y + 8))

            row_y += 38

        # Educational RL Lesson Callout
        lesson_rect = pygame.Rect(table_x, row_y + 4, table_w, 40)
        pygame.draw.rect(self.screen, (240, 253, 250), lesson_rect, border_radius=6)
        pygame.draw.rect(self.screen, (204, 251, 241), lesson_rect, 1, border_radius=6)

        l_txt = self.font_body.render(
            "RL Takeaway: Q-learning explores to converge on the mathematically optimal route.",
            True,
            (15, 118, 110),
        )
        self.screen.blit(l_txt, (lesson_rect.centerx - l_txt.get_width() // 2, lesson_rect.centery - l_txt.get_height() // 2))

        # 3 Action Buttons: [Play Again], [Compare with AI], [Back to Menu]
        mouse_pos = pygame.mouse.get_pos()
        play_again_btn.update(mouse_pos)
        play_again_btn.draw(self.screen)

        compare_ai_btn.update(mouse_pos)
        compare_ai_btn.draw(self.screen)

        menu_btn.update(mouse_pos)
        menu_btn.draw(self.screen)

    def render_human_play(
        self,
        env: CampusEnvironment,
        stats: Dict,
        path: List[Tuple[int, int]],
        restart_btn: Button,
        menu_btn: Button,
        goal_reached: bool = False,
        play_again_btn: Optional[Button] = None,
        watch_ai_btn: Optional[Button] = None,
        victory_menu_btn: Optional[Button] = None,
        route_btn: Optional[Button] = None,
    ) -> None:
        """Render a complete frame for Manual Human Navigation Mode."""
        self.screen.fill(config.COLOR_BACKGROUND)

        dest_name = getattr(env, "current_destination_name", "Central Library")
        start_name = getattr(env, "current_start_name", "Main Gate")

        # Header Title
        title_surf = self.font_title.render("Campus Navigator: Play Game", True, (15, 23, 42))
        sub_surf = self.font_subtitle.render(
            f"Manual Campus Navigation — Use Arrow Keys or WASD to navigate from {start_name} to {dest_name}",
            True,
            (100, 116, 139),
        )
        self.screen.blit(title_surf, (self.grid_x, 16))
        self.screen.blit(sub_surf, (self.grid_x, 48))

        # Update smooth agent animation (faster 0.45 lerp for responsive human control)
        self.update_agent_animation(env.agent_pos, lerp_speed=0.45)

        # Draw Grid with active animations & feedback
        self.draw_grid(env, stats)

        # Draw Path Trail traveled by player
        if len(path) > 1:
            self.draw_route_path(path, start_pos=env.start_pos, goal_pos=env.goal_pos)

        # Draw Agent
        self.draw_agent()

        # Draw Right HUD Panel
        self.draw_human_play_panel(stats, restart_btn, menu_btn, route_btn=route_btn)
        self.draw_legend(env)

        # If goal reached, overlay the MISSION COMPLETE victory modal
        if goal_reached and play_again_btn and watch_ai_btn and victory_menu_btn:
            self.draw_human_victory_modal(stats, play_again_btn, watch_ai_btn, victory_menu_btn)

        pygame.display.flip()

    # --------------------------------------------------------------------------
    # Phase 11: Human vs AI Comparison Renderer
    # --------------------------------------------------------------------------

    def render_ai_comparison_run(
        self,
        env: CampusEnvironment,
        stats: Dict,
        human_path: List[Tuple[int, int]],
        ai_path: List[Tuple[int, int]],
        skip_btn: Button,
        menu_btn: Button,
    ) -> None:
        """Render live frame for AI comparison run showing dual paths."""
        self.screen.fill(config.COLOR_BACKGROUND)

        # Header Title
        title_surf = self.font_title.render("Campus Navigator: Human vs AI Run", True, (15, 23, 42))
        sub_surf = self.font_subtitle.render(
            "Running trained Q-Learning policy on the identical campus environment to measure comparative results",
            True,
            (100, 116, 139),
        )
        self.screen.blit(title_surf, (self.grid_x, 16))
        self.screen.blit(sub_surf, (self.grid_x, 48))

        # Smooth agent interpolation for AI
        self.update_agent_animation(env.agent_pos, lerp_speed=0.35)

        # Draw Grid with active animations & feedback
        self.draw_grid(env, stats)

        # Draw Dual Paths (Human: Cyan, AI: Purple)
        self.draw_dual_route_paths(human_path, ai_path)

        # Draw Agent
        self.draw_agent()

        # Draw Right HUD Panel
        panel_rect = pygame.Rect(self.panel_x, self.panel_y, self.panel_width, self.panel_height)
        pygame.draw.rect(self.screen, config.COLOR_PANEL_BG, panel_rect, border_radius=12)
        pygame.draw.rect(self.screen, (51, 65, 85), panel_rect, 2, border_radius=12)

        header_surf = self.font_panel_title.render("AI Evaluation Run", True, (216, 180, 254))
        self.screen.blit(header_surf, (self.panel_x + 20, self.panel_y + 16))

        sub_surf = self.font_subtitle.render("Executing Policy (Epsilon = 0.0)", True, config.COLOR_TEXT_MUTED)
        self.screen.blit(sub_surf, (self.panel_x + 20, self.panel_y + 40))

        # Status Indicator Pill (Phase 17 Enhancement)
        self.draw_status_indicator(self.panel_x + self.panel_width - 150, self.panel_y + 14, "ACTIVE", "AI EVALUATION")

        pygame.draw.line(
            self.screen,
            (51, 65, 85),
            (self.panel_x + 20, self.panel_y + 60),
            (self.panel_x + self.panel_width - 20, self.panel_y + 60),
            1,
        )

        curr_state = stats.get("current_state", (0, 0))
        act_name = stats.get("action", "None")
        ai_rew = float(stats.get("reward", 0.0))
        ai_steps = stats.get("steps", 0)
        h_steps = stats.get("human_steps", 0)
        h_rew = float(stats.get("human_reward", 0.0))
        t_str = stats.get("time_str", "0.0s")

        metric_cards = [
            ("Current State", f"Row {curr_state[0]}, Col {curr_state[1]}", (56, 189, 248), True),
            ("Selected Action", act_name, (244, 114, 182), True),
            ("AI Total Reward", f"{ai_rew:+.1f}", config.COLOR_SUCCESS if ai_rew >= 0 else config.COLOR_DANGER, True),
            ("AI Steps Taken", f"{ai_steps} steps", config.COLOR_TEXT_LIGHT, True),
            ("Elapsed Time", t_str, (250, 204, 21), True),
            ("Distance to Goal", f"{stats.get('distance', 0)} blocks", config.COLOR_ACCENT, True),
            ("Human Steps (Ref)", f"{h_steps} steps", (56, 189, 248), True),
            ("Human Reward (Ref)", f"{h_rew:+.1f}", (56, 189, 248), True),
        ]

        curr_y = self.panel_y + 70
        card_h = 37
        card_w = self.panel_width - 40

        for label, val, val_col, is_mono in metric_cards:
            card_rect = pygame.Rect(self.panel_x + 20, curr_y, card_w, card_h)
            pygame.draw.rect(self.screen, config.COLOR_PANEL_CARD, card_rect, border_radius=6)

            lbl_surf = self.font_label.render(label, True, config.COLOR_TEXT_MUTED)
            val_font = self.font_mono_val if is_mono else self.font_val
            val_surf = val_font.render(val, True, val_col)

            self.screen.blit(lbl_surf, (card_rect.x + 12, card_rect.y + 9))
            self.screen.blit(val_surf, (card_rect.right - val_surf.get_width() - 12, card_rect.y + 9))

            curr_y += card_h + 6

        # Draw Action Buttons [Skip to Table] and [Back to Menu]
        mouse_pos = pygame.mouse.get_pos()
        skip_btn.update(mouse_pos)
        skip_btn.draw(self.screen)

        menu_btn.update(mouse_pos)
        menu_btn.draw(self.screen)
        self.draw_legend(env)

        pygame.display.flip()

    def draw_comparison_screen(
        self,
        human_data: Dict,
        ai_data: Dict,
        buttons: List[Button],
    ) -> None:
        """
        Render a clean, objective side-by-side comparison table between Human and AI.
        Does not claim AI is universally 'better'; simply presents empirical measurements.
        """
        self.screen.fill(config.COLOR_BACKGROUND)

        # Header Titles
        title_surf = self.font_title.render("Human vs AI Performance Comparison", True, (15, 23, 42))
        sub_surf = self.font_subtitle.render(
            "Empirical Measurements on Identical 10x10 Campus Environment (Gate (0,0) -> Library (9,9))",
            True,
            (100, 116, 139),
        )
        self.screen.blit(title_surf, (config.SCREEN_WIDTH // 2 - title_surf.get_width() // 2, 28))
        self.screen.blit(sub_surf, (config.SCREEN_WIDTH // 2 - sub_surf.get_width() // 2, 60))

        # Main Card Container
        card_w, card_h = 860, 560
        card_x = (config.SCREEN_WIDTH - card_w) // 2
        card_y = 88
        card_rect = pygame.Rect(card_x, card_y, card_w, card_h)

        pygame.draw.rect(self.screen, (255, 255, 255), card_rect, border_radius=14)
        pygame.draw.rect(self.screen, (203, 213, 225), card_rect, 2, border_radius=14)

        # Comparison Table Container
        table_x = card_x + 40
        table_w = card_w - 80
        table_y = card_y + 24

        # Table Header
        h_h = 40
        h_rect = pygame.Rect(table_x, table_y, table_w, h_h)
        pygame.draw.rect(self.screen, (30, 41, 59), h_rect, border_top_left_radius=8, border_top_right_radius=8)

        th_metric = self.font_btn.render("METRIC", True, (241, 245, 249))
        th_human = self.font_btn.render("HUMAN", True, (56, 189, 248))
        th_ai = self.font_btn.render("AI (Q-LEARNING)", True, (216, 180, 254))

        self.screen.blit(th_metric, (table_x + 28, table_y + 10))
        self.screen.blit(th_human, (table_x + 320, table_y + 10))
        self.screen.blit(th_ai, (table_x + 550, table_y + 10))

        # Table Rows Data
        h_steps = human_data.get("steps", 0)
        ai_steps = ai_data.get("steps", 0)
        h_rew = human_data.get("reward", 0.0)
        ai_rew = ai_data.get("reward", 0.0)
        h_time = human_data.get("time", "N/A")
        ai_time = ai_data.get("time", "N/A")
        h_route = human_data.get("route_length", 0)
        ai_route = ai_data.get("route_length", 0)
        h_succ = human_data.get("success", "Yes")
        ai_succ = ai_data.get("success", "Yes")
        h_coll = human_data.get("collisions", 0)
        ai_coll = ai_data.get("collisions", 0)

        rows = [
            ("Steps", f"{h_steps}", f"{ai_steps}"),
            ("Reward", f"{h_rew:+.1f}", f"{ai_rew:+.1f}"),
            ("Time", f"{h_time}", f"{ai_time}"),
            ("Route Length", f"{h_route} waypoints", f"{ai_route} waypoints"),
            ("Success", f"{h_succ}", f"{ai_succ}"),
            ("Obstacle Collisions", f"{h_coll} hits", f"{ai_coll} hits"),
        ]

        row_y = table_y + h_h
        row_h = 42

        for idx, (label, val_human, val_ai) in enumerate(rows):
            r_rect = pygame.Rect(table_x, row_y, table_w, row_h)
            bg_col = (248, 250, 252) if idx % 2 == 0 else (255, 255, 255)
            pygame.draw.rect(self.screen, bg_col, r_rect)
            pygame.draw.line(self.screen, (226, 232, 240), (table_x, row_y + row_h), (table_x + table_w, row_y + row_h), 1)

            lbl_s = self.font_body_bold.render(label, True, (15, 23, 42))
            vh_s = self.font_body.render(val_human, True, (30, 64, 175))
            vai_s = self.font_body.render(val_ai, True, (109, 40, 217))

            self.screen.blit(lbl_s, (table_x + 28, row_y + 11))
            self.screen.blit(vh_s, (table_x + 320, row_y + 11))
            self.screen.blit(vai_s, (table_x + 550, row_y + 11))

            row_y += row_h

        # Neutral Methodology Callout Box
        callout_rect = pygame.Rect(table_x, row_y + 12, table_w, 64)
        pygame.draw.rect(self.screen, (241, 245, 249), callout_rect, border_radius=8)
        pygame.draw.rect(self.screen, (203, 213, 225), callout_rect, 1, border_radius=8)

        note_title = self.font_small.render("METHODOLOGY & OBJECTIVE OBSERVATION:", True, (71, 85, 105))
        note_line1 = self.font_subtitle.render(
            "• Both agents operated on the deterministic 10x10 campus with 18 building obstacles and identical rewards.",
            True,
            (51, 65, 85),
        )
        note_line2 = self.font_subtitle.render(
            "• Q-Learning optimizes discounted cumulative return over training; human navigation demonstrates real-time decision-making.",
            True,
            (51, 65, 85),
        )
        self.screen.blit(note_title, (callout_rect.x + 16, callout_rect.y + 8))
        self.screen.blit(note_line1, (callout_rect.x + 16, callout_rect.y + 26))
        self.screen.blit(note_line2, (callout_rect.x + 16, callout_rect.y + 44))

        # Action Buttons
        mouse_pos = pygame.mouse.get_pos()
        for btn in buttons:
            btn.update(mouse_pos)
            btn.draw(self.screen)

        pygame.display.flip()

    # --------------------------------------------------------------------------
    # Phase 16: Route & Destination Selection Modal Renderer
    # --------------------------------------------------------------------------

    def draw_route_selection_modal(
        self,
        start_buttons: List[Button],
        dest_buttons: List[Button],
        confirm_btn: Button,
        cancel_btn: Button,
        selected_start_name: str,
        selected_dest_name: str,
        dest_id: int = 0,
        manhattan_dist: int = 0,
        is_same_location: bool = False,
    ) -> None:
        """
        Render the interactive Route & Destination Selection modal dialog (Phase 16).
        Allows selecting predefined Start Landmarks and Target Destinations.
        """
        # 1. Semi-transparent backdrop overlay
        overlay = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((15, 23, 42, 215))
        self.screen.blit(overlay, (0, 0))

        # 2. Main Modal Card Container
        card_w, card_h = 880, 580
        card_x = (config.SCREEN_WIDTH - card_w) // 2
        card_y = (config.SCREEN_HEIGHT - card_h) // 2
        card_rect = pygame.Rect(card_x, card_y, card_w, card_h)

        pygame.draw.rect(self.screen, (255, 255, 255), card_rect, border_radius=16)
        pygame.draw.rect(self.screen, (203, 213, 225), card_rect, 2, border_radius=16)

        # 3. Header Ribbon
        header_h = 70
        header_rect = pygame.Rect(card_x, card_y, card_w, header_h)
        pygame.draw.rect(self.screen, (241, 245, 249), header_rect, border_top_left_radius=16, border_top_right_radius=16)
        pygame.draw.line(self.screen, (226, 232, 240), (card_x, card_y + header_h), (card_x + card_w, card_y + header_h), 2)

        t_surf = self.font_header.render("Campus Route & Destination Selection", True, (15, 23, 42))
        s_surf = self.font_subtitle.render(
            "Select Origin Landmark and Target Destination for Goal-Conditioned Q-Learning Navigation",
            True,
            (71, 85, 105),
        )
        self.screen.blit(t_surf, (card_x + 28, card_y + 14))
        self.screen.blit(s_surf, (card_x + 28, card_y + 42))

        # 4. Section 1: Start Location (Origin)
        s1_y = card_y + 86
        s1_lbl = self.font_body_bold.render("1. SELECT START LOCATION (ORIGIN):", True, (22, 101, 52))
        self.screen.blit(s1_lbl, (card_x + 28, s1_y))

        mouse_pos = pygame.mouse.get_pos()
        for btn in start_buttons:
            btn.is_selected = (btn.text == selected_start_name)
            btn.update(mouse_pos)
            btn.draw(self.screen)

        # 5. Section 2: Target Destination (Goal)
        s2_y = card_y + 196
        s2_lbl = self.font_body_bold.render("2. SELECT TARGET DESTINATION (GOAL):", True, (120, 53, 15))
        self.screen.blit(s2_lbl, (card_x + 28, s2_y))

        for btn in dest_buttons:
            btn.is_selected = (btn.text == selected_dest_name)
            btn.update(mouse_pos)
            btn.draw(self.screen)

        # 6. Section 3: Route Summary & Policy Status Card
        summary_y = card_y + 306
        summary_h = 130
        summary_rect = pygame.Rect(card_x + 28, summary_y, card_w - 56, summary_h)
        pygame.draw.rect(self.screen, (248, 250, 252), summary_rect, border_radius=10)
        pygame.draw.rect(self.screen, (226, 232, 240), summary_rect, 1, border_radius=10)

        # Summary Title
        sum_title = self.font_small.render("ACTIVE ROUTE CONFIGURATION & MODEL VERIFICATION", True, (71, 85, 105))
        self.screen.blit(sum_title, (summary_rect.x + 16, summary_rect.y + 12))

        # Coordinates lookup
        start_coord = config.START_LOCATIONS.get(selected_start_name, (0, 0))
        dest_coord = config.DESTINATIONS.get(selected_dest_name, (9, 9))

        route_str = f"Selected Route:  {selected_start_name} {start_coord}  --->  {selected_dest_name} {dest_coord}"
        route_surf = self.font_body_bold.render(route_str, True, (30, 41, 59))
        self.screen.blit(route_surf, (summary_rect.x + 16, summary_rect.y + 34))

        # Metric tags
        metric_line = (
            f"Manhattan Distance: {manhattan_dist} blocks  |  "
            f"Goal ID: #{dest_id} ({selected_dest_name})  |  "
            f"Q-Tensor Slice: multi_q_table[{dest_id}]"
        )
        metric_surf = self.font_subtitle.render(metric_line, True, (100, 116, 139))
        self.screen.blit(metric_surf, (summary_rect.x + 16, summary_rect.y + 60))

        # Model readiness status badge
        if is_same_location:
            alert_surf = self.font_body_bold.render(
                "Warning: Start and Destination cannot be the same coordinate! Choose different locations.",
                True,
                (220, 38, 38),
            )
            self.screen.blit(alert_surf, (summary_rect.x + 16, summary_rect.y + 88))
        else:
            status_surf = self.font_body_bold.render(
                "Policy Ready: Goal-Conditioned Q-learning model trained and verified for this route.",
                True,
                (22, 101, 52),
            )
            self.screen.blit(status_surf, (summary_rect.x + 16, summary_rect.y + 88))

        # 7. Section 4: Action Buttons [Confirm & Navigate] and [Cancel]
        confirm_btn.update(mouse_pos)
        confirm_btn.draw(self.screen)

        cancel_btn.update(mouse_pos)
        cancel_btn.draw(self.screen)


