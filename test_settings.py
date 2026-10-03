"""
Standalone Test Suite for Phase 14: Settings Configuration & Persistence
========================================================================
File: test_settings.py

Validates:
1. Factory default configuration retrieval.
2. Comprehensive algorithmic validation and rejection of invalid values:
   - Negative learning rate
   - Discount factor > 1
   - Negative episodes
   - Zero actions
   - Min epsilon > Initial epsilon
   - Invalid epsilon decay
   - Non-positive max steps
   - Unsupported grid size
   - Invalid training speed
3. Persistence and loading from config.json.
4. Reset to defaults functionality.
5. Headless GUI rendering and input handling of the Settings sub-screen.
"""

import json
import os
import sys

# Set SDL to dummy video driver for headless automated test
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import config
from game.menu import GameMenu
from game.game import CampusNavigatorGame


def test_settings_validation_and_persistence():
    print("=" * 75)
    print("       TEST 1: CONFIGURATION VALIDATION & PERSISTENCE (config.json)      ")
    print("=" * 75)

    # 1. Test Default Configuration
    defaults = config.get_default_config()
    print("[1] Verifying factory default configuration values...")
    assert defaults["grid_size"] == 10
    assert defaults["training_episodes"] == 1000
    assert defaults["learning_rate"] == 0.1
    assert defaults["discount_factor"] == 0.9
    assert defaults["initial_epsilon"] == 1.0
    assert defaults["min_epsilon"] == 0.05
    assert defaults["epsilon_decay"] == 0.995
    assert defaults["max_steps_per_episode"] == 200
    assert defaults["training_speed"] == "NORMAL"
    assert defaults["num_actions"] == 4
    assert defaults["reward_safe_zone"] == 2.0
    assert defaults["reward_crowded_zone"] == -2.0
    assert defaults["reward_unsafe_zone"] == -10.0

    valid, msg = config.validate_config(defaults)
    assert valid, f"Default configuration failed validation: {msg}"
    print(f"    [+] Default configuration is valid: {msg}")

    # 2. Test Invalid Values Rejection
    print("\n[2] Testing algorithmic validation guardrails (explicit rejection rules)...")

    # A. Negative Learning Rate
    cfg_neg_lr = dict(defaults, learning_rate=-0.05)
    valid, msg = config.validate_config(cfg_neg_lr)
    assert not valid, "Failed to reject negative learning rate!"
    print(f"    [+] Successfully rejected negative learning rate (-0.05): {msg}")

    # B. Learning Rate Zero
    cfg_zero_lr = dict(defaults, learning_rate=0.0)
    valid, msg = config.validate_config(cfg_zero_lr)
    assert not valid, "Failed to reject zero learning rate!"
    print(f"    [+] Successfully rejected zero learning rate (0.0): {msg}")

    # C. Discount Factor > 1
    cfg_high_df = dict(defaults, discount_factor=1.25)
    valid, msg = config.validate_config(cfg_high_df)
    assert not valid, "Failed to reject discount factor > 1!"
    print(f"    [+] Successfully rejected discount factor > 1 (1.25): {msg}")

    # D. Discount Factor Negative
    cfg_neg_df = dict(defaults, discount_factor=-0.1)
    valid, msg = config.validate_config(cfg_neg_df)
    assert not valid, "Failed to reject negative discount factor!"
    print(f"    [+] Successfully rejected negative discount factor (-0.1): {msg}")

    # E. Negative Training Episodes
    cfg_neg_ep = dict(defaults, training_episodes=-500)
    valid, msg = config.validate_config(cfg_neg_ep)
    assert not valid, "Failed to reject negative episodes!"
    print(f"    [+] Successfully rejected negative episodes (-500): {msg}")

    # F. Zero Actions
    cfg_zero_act = dict(defaults, num_actions=0)
    valid, msg = config.validate_config(cfg_zero_act)
    assert not valid, "Failed to reject zero actions!"
    print(f"    [+] Successfully rejected zero actions (0): {msg}")

    # G. Min Epsilon > Initial Epsilon
    cfg_inv_eps = dict(defaults, initial_epsilon=0.1, min_epsilon=0.5)
    valid, msg = config.validate_config(cfg_inv_eps)
    assert not valid, "Failed to reject min_epsilon > initial_epsilon!"
    print(f"    [+] Successfully rejected min_epsilon > initial_epsilon (0.5 > 0.1): {msg}")

    # H. Invalid Epsilon Decay
    cfg_inv_decay = dict(defaults, epsilon_decay=1.05)
    valid, msg = config.validate_config(cfg_inv_decay)
    assert not valid, "Failed to reject epsilon decay > 1!"
    print(f"    [+] Successfully rejected epsilon decay > 1 (1.05): {msg}")

    # I. Invalid Max Steps (Zero or Negative)
    cfg_zero_steps = dict(defaults, max_steps_per_episode=0)
    valid, msg = config.validate_config(cfg_zero_steps)
    assert not valid, "Failed to reject zero max steps!"
    print(f"    [+] Successfully rejected zero max steps (0): {msg}")

    # J. Invalid Training Speed
    cfg_inv_speed = dict(defaults, training_speed="TURBO_SPEED")
    valid, msg = config.validate_config(cfg_inv_speed)
    assert not valid, "Failed to reject invalid speed string!"
    print(f"    [+] Successfully rejected invalid speed ('TURBO_SPEED'): {msg}")

    # K. Unsupported Grid Size (< 6 or > 16)
    cfg_inv_grid = dict(defaults, grid_size=4)
    valid, msg = config.validate_config(cfg_inv_grid)
    assert not valid, "Failed to reject unsupported grid size!"
    print(f"    [+] Successfully rejected unsupported grid size (4x4): {msg}")

    # L. Negative Safe Zone Reward Bonus (< 0)
    cfg_neg_safe = dict(defaults, reward_safe_zone=-5.0)
    valid, msg = config.validate_config(cfg_neg_safe)
    assert not valid, "Failed to reject negative safe zone reward!"
    print(f"    [+] Successfully rejected negative safe zone reward (-5.0): {msg}")

    # M. Positive Crowded Zone Penalty (> 0)
    cfg_pos_crowd = dict(defaults, reward_crowded_zone=3.0)
    valid, msg = config.validate_config(cfg_pos_crowd)
    assert not valid, "Failed to reject positive crowded zone penalty!"
    print(f"    [+] Successfully rejected positive crowded zone penalty (+3.0): {msg}")

    # N. Positive Unsafe Zone Penalty (> 0)
    cfg_pos_unsafe = dict(defaults, reward_unsafe_zone=10.0)
    valid, msg = config.validate_config(cfg_pos_unsafe)
    assert not valid, "Failed to reject positive unsafe zone penalty!"
    print(f"    [+] Successfully rejected positive unsafe zone penalty (+10.0): {msg}")

    # 3. Test Save & Load from config.json
    print("\n[3] Testing persistence to and loading from config.json...")
    test_cfg = dict(defaults, learning_rate=0.15, training_episodes=800, training_speed="FAST")
    ok, save_msg = config.save_config(test_cfg)
    assert ok, f"Failed to save config: {save_msg}"
    assert os.path.exists(config.CONFIG_PATH), "config.json was not created on disk!"
    print(f"    [+] Successfully saved custom settings to {config.CONFIG_PATH}")

    # Verify JSON content on disk
    with open(config.CONFIG_PATH, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert loaded_json["learning_rate"] == 0.15
    assert loaded_json["training_episodes"] == 800
    assert loaded_json["training_speed"] == "FAST"
    print("    [+] On-disk JSON matches saved dictionary perfectly.")

    # Verify module globals updated
    assert config.LEARNING_RATE == 0.15
    assert config.DEFAULT_TRAINING_EPISODES == 800
    assert config.DEFAULT_TRAINING_SPEED == "FAST"
    print("    [+] Active module globals dynamically updated in config.py.")

    # 4. Test Reset to Defaults
    print("\n[4] Testing reset to factory defaults...")
    restored = config.reset_to_defaults()
    assert restored["learning_rate"] == 0.1
    assert restored["training_episodes"] == 1000
    assert config.LEARNING_RATE == 0.1
    assert config.DEFAULT_TRAINING_EPISODES == 1000
    print("    [+] Factory defaults restored and persisted successfully.")


def test_settings_gui_headless():
    print("\n" + "=" * 75)
    print("          TEST 2: SETTINGS SCREEN HEADLESS GUI & EVENT HANDLING         ")
    print("=" * 75)

    pygame.init()
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))

    callback_called = [False]

    def on_saved():
        callback_called[0] = True

    menu = GameMenu(screen, on_settings_saved=on_saved)

    # 1. Switch to SETTINGS sub-screen
    print("[1] Switching to SETTINGS sub-screen...")
    menu.sub_screen = "SETTINGS"
    menu._load_settings_into_form()
    assert menu.sub_screen == "SETTINGS"
    print("    [+] Switched to SETTINGS mode.")

    # 2. Render Settings Screen
    print("[2] Rendering Settings Screen with Pygame...")
    menu.draw()
    print("    [+] Settings Screen rendered without errors.")

    # 3. Simulate Stepper Click: Increment Learning Rate
    print("[3] Simulating stepper increment on learning_rate...")
    initial_lr = menu.settings_data["learning_rate"]
    plus_data = menu.settings_rects["learning_rate_plus"]
    step = plus_data[3]
    dec = plus_data[6]
    menu.settings_data["learning_rate"] = round(initial_lr + step, dec)
    assert menu.settings_data["learning_rate"] == round(initial_lr + 0.01, 2)
    print(f"    [+] Learning rate adjusted from {initial_lr} to {menu.settings_data['learning_rate']}")

    # 4. Simulate Save Settings Click
    print("[4] Simulating [Save Settings] button execution...")
    menu._save_settings_from_form()
    assert callback_called[0], "on_settings_saved callback was not triggered!"
    assert menu.settings_status[0] == "success"
    print(f"    [+] Settings saved. Status: {menu.settings_status[1]}")

    # 5. Simulate Invalid Input through Text Buffer
    print("[5] Simulating invalid text input (negative value)...")
    menu.settings_active_field = "learning_rate"
    menu.settings_text_buffer = "-0.25"
    menu._save_settings_from_form()
    assert menu.settings_status[0] == "error"
    print(f"    [+] Rejection successfully caught: {menu.settings_status[1]}")

    # 6. Simulate Reset Defaults
    print("[6] Simulating [Reset Defaults] button execution...")
    callback_called[0] = False
    menu._reset_settings_to_defaults()
    assert callback_called[0], "on_settings_saved callback was not triggered on reset!"
    assert menu.settings_status[0] == "info"
    assert menu.settings_data["learning_rate"] == 0.1
    print("    [+] Reset Defaults executed cleanly.")

    # 7. Test Integration with CampusNavigatorGame instance
    print("\n[7] Testing CampusNavigatorGame integration with Settings...")
    game = CampusNavigatorGame()
    assert game.menu is not None
    assert game.menu.option_labels[5][0] == "Settings"
    assert game.menu.option_labels[6][0] == "Exit"
    print("    [+] Main Menu has option [6] Settings and [7] Exit configured.")

    print("\n" + "=" * 75)
    print("             ALL PHASE 14 SETTINGS TESTS PASSED SUCCESSFULLY!           ")
    print("=" * 75)


if __name__ == "__main__":
    test_settings_validation_and_persistence()
    test_settings_gui_headless()
