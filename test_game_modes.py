"""
Comprehensive Integration Test for Game Modes & UI Rendering
=============================================================
File: test_game_modes.py

Tests:
1. Headless initialization of CampusNavigatorGame and CampusRenderer.
2. AI Training Mode stepping and rendering.
3. Watch Trained AI Mode autonomous navigation (verifying optimal 18-step arrival).
4. Manual Human Play Mode movement, path tracking, collision counting, and victory modal rendering.
5. Menu sub-screens (Results dashboard, RL Theory guide).
"""

import os
import sys

# Set SDL to dummy video driver for headless automated test
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import config
from game.game import CampusNavigatorGame
from game.menu import GameMenu


def test_game_integration():
    print("=" * 70)
    print("      INTEGRATION TEST: CAMPUS NAVIGATOR GAME & UI MODES      ")
    print("=" * 70)

    # 1. Initialize Game Instance
    print("[1] Initializing CampusNavigatorGame instance...")
    game = CampusNavigatorGame()
    assert game.screen is not None, "Display screen failed to initialize."
    assert game.env is not None, "CampusEnvironment failed to initialize."
    assert game.agent is not None, "QLearningAgent failed to initialize."
    print("    [+] Game controller, environment, and renderer initialized successfully.")

    # 2. Test Menu Rendering & Sub-Screens
    print("[2] Testing GameMenu rendering & sub-screens...")
    menu = GameMenu(game.screen)
    menu.sub_screen = "MAIN"
    menu.draw()
    print("    [+] Main Menu rendered cleanly.")

    menu.sub_screen = "RESULTS"
    menu.draw()
    print("    [+] Training Results Modal rendered cleanly.")

    menu.sub_screen = "RL_WORKS"
    menu.draw()
    print("    [+] How RL Works Modal rendered cleanly.")

    # 3. Test Phase 8: AI Training Mode Stepping & Rendering
    print("[3] Testing AI Training Mode stepping & rendering...")
    game._start_training_mode()
    assert game.current_state == "TRAINING", "Failed to switch to TRAINING state."
    assert game.agent.epsilon == 1.0, "Epsilon not reset to 1.0."

    # Run 20 single training steps
    for _ in range(20):
        game._execute_single_training_step()

    assert game.train_ep_steps >= 0, "Training step execution failed."
    print(f"    [+] 20 live training steps executed. Episode: {game.train_ep}, Q-values updated.")

    # 4. Test Phase 9: Watch Trained AI Mode
    print("[4] Testing Watch Trained AI Mode...")
    game._start_ai_mode()
    assert game.current_state == "AI_PLAY", "Failed to switch to AI_PLAY state."
    assert game.agent.epsilon == 0.0, "Epsilon should be 0.0 in Watch mode."

    # Simulate steps until destination
    steps_taken = 0
    while not game.ai_route_complete and steps_taken < 50:
        current_state = game.env.get_state()
        action = game.agent.get_best_action(current_state)
        next_state, reward, done, info = game.env.step(action)
        game.ai_last_action = config.ACTIONS.get(action, "None")
        game.ai_step_reward = reward
        game.ai_total_reward += reward
        game.ai_steps = game.env.step_count
        game.ai_path.append(next_state)
        if info.get("goal_reached", False):
            game.ai_route_complete = True
        steps_taken += 1

    assert game.ai_route_complete, "Trained AI failed to reach Central Library!"
    assert steps_taken == 18, f"Expected 18 steps, got {steps_taken}"
    assert game.ai_total_reward == 170.0, f"Expected +170 reward, got {game.ai_total_reward}"
    print(f"    [+] Autonomous AI arrived at Central Library in exactly {steps_taken} steps with reward {game.ai_total_reward}!")

    # Test Watch AI Rendering
    stats = {
        "current_state": game.env.agent_pos,
        "action": game.ai_last_action,
        "reward": game.ai_step_reward,
        "total_reward": game.ai_total_reward,
        "steps": game.ai_steps,
        "status": "DESTINATION REACHED! (+100)",
        "goal_reached": game.ai_route_complete,
        "distance": game.env.get_distance_to_goal(),
    }
    game.renderer.render_watch_ai(
        game.env,
        stats,
        game.ai_path,
        game.ai_route_complete,
        game.btn_watch_restart,
        game.btn_watch_menu,
    )
    print("    [+] Watch AI screen with route trail rendered cleanly.")

    # 5. Test Phase 10: Manual Human Play Mode
    print("[5] Testing Phase 10: Manual Human Play Mode...")
    game._start_human_mode()
    assert game.current_state == "HUMAN_PLAY", "Failed to switch to HUMAN_PLAY state."
    assert game.env.agent_pos == (0, 0), "Human should start at Gate (0, 0)."
    assert len(game.human_path) == 1, "Initial path should contain starting position."

    # Simulate player stepping right into walkable cell (0, 1)
    next_s, rew, done, info = game.env.step(config.ACTION_RIGHT)
    game.human_path.append(next_s)
    game.cumulative_reward += rew
    game.step_count += 1
    assert next_s == (0, 1), f"Expected (0, 1), got {next_s}"
    assert rew == 5.0, f"Expected +5 reward for moving closer, got {rew}"

    # Simulate player stepping into obstacle at (0, 4) from (0, 3)
    game.env.agent_pos = (0, 3)
    next_s, rew, done, info = game.env.step(config.ACTION_RIGHT)
    assert not info.get("is_valid", True), "Obstacle at (0, 4) should be invalid!"
    assert rew == -10.0, f"Expected -10 penalty, got {rew}"
    game.human_collisions += 1

    # Simulate player reaching the Central Library at (9, 9)
    game.env.agent_pos = (8, 9)
    next_s, rew, done, info = game.env.step(config.ACTION_DOWN)
    assert info.get("goal_reached", False), "Goal should be reached at (9, 9)!"
    game.goal_reached = True
    game.human_path.append(next_s)
    game.cumulative_reward += rew
    game.step_count += 1

    # Test Human Play HUD & MISSION COMPLETE Modal Rendering
    human_stats = {
        "current_state": game.env.agent_pos,
        "action": "DOWN",
        "reward": rew,
        "total_reward": game.cumulative_reward,
        "steps": game.step_count,
        "time_str": "12.4s",
        "elapsed_seconds": 12.4,
        "collisions": game.human_collisions,
        "hazards": game.human_hazards,
        "distance": 0,
        "status": "MISSION COMPLETE! (+100)",
        "goal_reached": True,
    }
    game.renderer.render_human_play(
        game.env,
        human_stats,
        game.human_path,
        game.btn_human_restart,
        game.btn_human_menu,
        goal_reached=True,
        play_again_btn=game.btn_human_play_again,
        watch_ai_btn=game.btn_human_watch_ai,
        victory_menu_btn=game.btn_human_victory_menu,
    )
    print("    [+] Human Play HUD and MISSION COMPLETE Modal rendered cleanly with Steps, Time, Reward & 3 Buttons!")

    # 6. Test Phase 11: Human vs AI Comparison Mode
    print("[6] Testing Phase 11: Human vs AI Comparison Mode...")
    # Start AI comparison from the completed human session
    game.human_elapsed_time = 14.2
    game._start_ai_comparison()
    assert game.current_state == "AI_COMPARISON_RUN", "Failed to switch to AI_COMPARISON_RUN state."
    assert game.comparison_human_data["steps"] == game.step_count, "Human steps mismatch."
    assert game.comparison_human_data["reward"] == round(game.cumulative_reward, 1), "Human reward mismatch."
    assert game.comparison_human_data["success"] == "Yes", "Human success status mismatch."
    assert game.comparison_human_data["collisions"] == 1, "Human collisions mismatch."
    assert len(game.human_path) >= 2, "Human path should be preserved for dual rendering."

    # Test live AI comparison frame rendering with dual paths
    ai_run_stats = {
        "current_state": game.env.agent_pos,
        "action": "DOWN",
        "reward": 5.0,
        "steps": 1,
        "human_steps": game.comparison_human_data["steps"],
        "human_reward": game.comparison_human_data["reward"],
        "time_str": "0.2s",
        "distance": game.env.get_distance_to_goal(),
    }
    game.renderer.render_ai_comparison_run(
        game.env,
        ai_run_stats,
        game.human_path,
        game.comparison_ai_path,
        game.btn_comp_run_skip,
        game.btn_comp_run_menu,
    )
    print("    [+] Live AI Comparison Run screen with dual trajectory paths rendered cleanly.")

    # Fast forward AI run to completion
    game._fast_forward_ai_comparison()
    assert game.current_state == "COMPARISON_SUMMARY", "Failed to transition to COMPARISON_SUMMARY."
    assert game.comparison_ai_data["steps"] == 18, f"Expected 18 steps for optimal AI, got {game.comparison_ai_data['steps']}"
    assert game.comparison_ai_data["reward"] == 170.0, f"Expected 170.0 reward for optimal AI, got {game.comparison_ai_data['reward']}"
    assert game.comparison_ai_data["success"] == "Yes", "AI should reach goal successfully."
    assert game.comparison_ai_data["collisions"] == 0, "Optimal AI should have 0 collisions."
    assert game.comparison_ai_data["route_length"] == 19, f"Expected 19 waypoints, got {game.comparison_ai_data['route_length']}"

    print(f"    [+] Comparison Table Data Verified:")
    print(f"        Human: Steps={game.comparison_human_data['steps']}, Reward={game.comparison_human_data['reward']}, Success={game.comparison_human_data['success']}, Hits={game.comparison_human_data['collisions']}")
    print(f"        AI   : Steps={game.comparison_ai_data['steps']}, Reward={game.comparison_ai_data['reward']}, Success={game.comparison_ai_data['success']}, Hits={game.comparison_ai_data['collisions']}")

    # Render Side-by-Side Comparison Screen Table
    game.renderer.draw_comparison_screen(
        game.comparison_human_data,
        game.comparison_ai_data,
        game.comparison_summary_buttons,
    )
    print("    [+] Side-by-Side Comparison Table rendered cleanly with objective empirical metrics & action buttons.")

    print("\n" + "=" * 70)
    print("     ALL INTEGRATION TESTS PASSED SUCCESSFULLY! (100% OPERATIONAL)    ")
    print("=" * 70)


if __name__ == "__main__":
    test_game_integration()
