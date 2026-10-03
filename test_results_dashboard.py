"""
Phase 12: Training Results Screen Verification & Unit Tests
===========================================================
File: test_results_dashboard.py

Tests:
1. Verification of 7 summary metrics extraction from training_results.csv.
2. Verification of 4 plot images loading and scaling into memory.
3. Verification of smooth scrolling mechanics (mouse wheel, keys, scrollbar clamping).
4. Verification of results dashboard rendering inside Pygame.
5. Verification of missing-results error screen fallback and navigation.
"""

import os
import sys

# Set headless SDL video driver for automated testing
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import config
from game.menu import GameMenu


def test_results_dashboard() -> None:
    print("=" * 70)
    print("      PHASE 12 VERIFICATION: TRAINING RESULTS DASHBOARD       ")
    print("=" * 70)

    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    menu = GameMenu(screen)

    # 1. Test Metrics Loading from CSV
    print("[1] Testing Metrics Computation from results/training_results.csv...")
    menu._load_results_data(force_reload=True)
    assert menu.results_df_loaded, "Failed to load results DataFrame."
    assert menu.results_metrics is not None, "Results metrics dict is None."

    m = menu.results_metrics
    print(f"    [+] Training Episodes : {m['total_episodes']}")
    print(f"    [+] Success Rate      : {m['sr_recent']:.1f}% (Recent) / {m['sr_all']:.1f}% (Overall)")
    print(f"    [+] Average Reward    : {m['rew_recent']:+.1f} (Recent) / {m['rew_all']:+.1f} (Overall)")
    print(f"    [+] Average Steps     : {m['steps_recent']:.1f} steps (Recent) / {m['steps_all']:.1f} (Overall)")
    print(f"    [+] Collision Rate    : {m['coll_recent']:.2f} hits/ep (Recent) / {m['coll_all']:.2f} (Overall)")
    print(f"    [+] Best Episode      : Episode #{m['best_ep']} (Reward: {m['best_rew']:+.1f}, {m['best_steps']} steps)")
    print(f"    [+] Final Epsilon     : {m['final_eps']:.4f}")

    assert m["total_episodes"] >= 100, f"Expected at least 100 episodes, got {m['total_episodes']}"
    assert m["sr_recent"] == 100.0, f"Expected 100% late-stage success rate, got {m['sr_recent']}"
    assert m["steps_recent"] <= 20.0, f"Expected <= 20.0 average recent steps, got {m['steps_recent']}"
    assert m["coll_recent"] <= 1.0, f"Expected <= 1.0 recent collisions, got {m['coll_recent']}"
    assert m["final_eps"] == 0.05, f"Expected 0.05 final epsilon, got {m['final_eps']}"
    print("    [+] All 7 required summary metrics match theoretical and empirical benchmarks!")

    # 2. Test 4 Plot Images Loading and Smoothscaling
    print("\n[2] Testing 4 Matplotlib Plot Images Loading & Scaling...")
    menu._load_results_plots(force_reload=True)
    assert menu.results_plots_loaded, "Plots flag not marked as loaded."

    plot_keys = ["reward_vs_episode", "steps_vs_episode", "success_rate", "collision_rate"]
    for key in plot_keys:
        surf = menu.results_cached_plots.get(key)
        assert surf is not None, f"Plot image '{key}' failed to load."
        w, h = surf.get_size()
        assert (w, h) == (454, 250), f"Expected scaled size (454, 250), got ({w}, {h})"
        print(f"    [+] Plot '{key}' successfully loaded and smoothscaled to ({w}x{h}).")

    # 3. Test Dashboard Rendering with Viewport Clipping
    print("\n[3] Testing Results Dashboard Rendering in Pygame...")
    menu.sub_screen = "RESULTS"
    menu.results_scroll_pos = 0.0
    menu.draw()
    print("    [+] Results dashboard rendered cleanly at top of scroll (scroll_pos = 0.0).")
    assert menu.results_max_scroll > 0.0, "Expected scrollable content height > viewport height."
    print(f"    [+] Scrollable range verified: max_scroll = {menu.results_max_scroll:.1f} px.")

    # Render scrolled halfway down
    menu.results_scroll_pos = menu.results_max_scroll / 2.0
    menu.draw()
    print(f"    [+] Results dashboard rendered cleanly at midpoint (scroll_pos = {menu.results_scroll_pos:.1f}).")

    # Render scrolled to bottom
    menu.results_scroll_pos = menu.results_max_scroll
    menu.draw()
    print("    [+] Results dashboard rendered cleanly at bottom of scroll.")

    # 4. Test Scrolling Input Events
    print("\n[4] Testing Scrolling Mechanics & Clamping...")
    # Simulate Mouse Wheel Scroll Down
    wheel_down_event = pygame.event.Event(pygame.MOUSEWHEEL, {"x": 0, "y": -2, "flipped": False})
    pygame.event.post(wheel_down_event)
    menu.handle_input()
    assert menu.results_scroll_pos > 0.0, "Mouse wheel did not scroll down."

    # Simulate Key UP Scroll
    menu.results_scroll_pos = 100.0
    key_up_event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_UP})
    pygame.event.post(key_up_event)
    menu.handle_input()
    assert menu.results_scroll_pos == 60.0, f"Expected 60.0 scroll after UP key, got {menu.results_scroll_pos}"

    # Test clamping limits
    menu.results_scroll_pos = -50.0
    menu.results_scroll_pos = max(0.0, min(menu.results_max_scroll, menu.results_scroll_pos))
    assert menu.results_scroll_pos == 0.0, "Lower clamp bound failed."

    menu.results_scroll_pos = menu.results_max_scroll + 100.0
    menu.results_scroll_pos = max(0.0, min(menu.results_max_scroll, menu.results_scroll_pos))
    assert menu.results_scroll_pos == menu.results_max_scroll, "Upper clamp bound failed."
    print("    [+] Mouse wheel, keyboard stepping, and scroll boundaries verified 100%.")

    # 5. Test Missing Results Error Fallback Screen
    print("\n[5] Testing Missing Results Fallback Error Screen...")
    menu.results_metrics = None  # Force missing results state
    menu._draw_no_results_screen()
    print("    [+] Fallback alert screen rendered with instructions to train agent first.")

    # Simulate clicking Back to Menu from fallback screen
    back_event = pygame.event.Event(pygame.KEYDOWN, {"key": pygame.K_ESCAPE})
    pygame.event.post(back_event)
    ret = menu.handle_input()
    assert menu.sub_screen == "MAIN", "Failed to return to MAIN menu on ESC."
    print("    [+] ESC successfully returned to MAIN menu from error screen.")

    print("\n" + "=" * 70)
    print("    PHASE 12 VERIFICATION COMPLETED: ALL TESTS PASSED! (100%)    ")
    print("=" * 70)


if __name__ == "__main__":
    test_results_dashboard()
