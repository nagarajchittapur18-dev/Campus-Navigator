"""
Campus Navigator: Phase 1 Environment Test Script
=================================================
File: test_campus_env.py

Demonstrates:
1. Resetting the environment
2. Printing the starting state
3. Performing several actions
4. Displaying next state and reward
5. Detecting the goal
6. Demonstrating invalid movement handling
"""

import config
from environment.campus_env import CampusEnvironment


def test_campus_environment():
    print("======================================================================")
    print("        TESTING PHASE 1: CAMPUS ENVIRONMENT IMPLEMENTATION           ")
    print("======================================================================")

    # 1. Reset the environment
    env = CampusEnvironment()
    start_state = env.reset()
    print(f"\n[TEST 1 & 2] Reset & Starting State:")
    print(f"  Starting State        : {start_state}")
    print(f"  Target Destination    : {env.goal_pos}")
    print(f"  Initial Distance      : {env.get_distance_to_goal()} steps")
    print(f"  Grid Dimensions       : {env.rows} rows x {env.cols} columns")

    # Render starting ASCII map
    print("\n[ASCII CAMPUS MAP] (A: Agent, S: Gate, G: Goal, #: Building, Z: Safe, !: Unsafe)")
    print(env.render_ascii(show_agent=True))

    # 3 & 4. Perform several actions & display next state and reward
    print("\n[TEST 3 & 4] Performing Actions and Observing States & Rewards:")
    actions_to_test = [
        (config.ACTION_DOWN, "Move DOWN"),
        (config.ACTION_RIGHT, "Move RIGHT"),
        (config.ACTION_RIGHT, "Move RIGHT"),
        (config.ACTION_UP, "Move UP (Moving farther from goal)"),
    ]

    for action_code, action_name in actions_to_test:
        current_s = env.get_state()
        next_s, reward, done, info = env.step(action_code)
        print(
            f"  Action: {action_name:<32} | Transition: {current_s} -> {next_s} | "
            f"Reward: {reward:+5.1f} | Event: {info['event']:<15} | Done: {done}"
        )

    # 5. Invalid movement handling
    print("\n[TEST 5] Demonstrating Invalid Movement Handling:")
    # Reset back to (0, 0)
    env.reset()
    print(f"  Agent reset to: {env.get_state()}")

    # Out of bounds UP
    next_s, reward, done, info = env.step(config.ACTION_UP)
    print(f"  Attempt move UP out of bounds   -> State: {next_s} | Reward: {reward:+5.1f} | Valid: {info['is_valid']}")

    # Out of bounds LEFT
    next_s, reward, done, info = env.step(config.ACTION_LEFT)
    print(f"  Attempt move LEFT out of bounds -> State: {next_s} | Reward: {reward:+5.1f} | Valid: {info['is_valid']}")

    # Move right into obstacle at (0, 4)
    for _ in range(3):
        env.step(config.ACTION_RIGHT)
    print(f"  Agent navigated to {env.get_state()} (adjacent to Obstacle at (0, 4))")
    next_s, reward, done, info = env.step(config.ACTION_RIGHT)
    print(f"  Attempt move RIGHT into Obstacle -> State: {next_s} | Reward: {reward:+5.1f} | Valid: {info['is_valid']}")

    # 6. Detect the goal
    print("\n[TEST 6] Detecting Goal Arrival:")
    # Teleport right above goal at (8, 9)
    env.agent_pos = (8, 9)
    print(f"  Agent placed at {env.get_state()} (one step North of Goal {env.goal_pos})")
    next_s, reward, done, info = env.step(config.ACTION_DOWN)
    print(
        f"  Stepped DOWN into Goal (9, 9)   -> State: {next_s} | Reward: {reward:+5.1f} | "
        f"Goal Reached: {info['goal_reached']} | Terminal: {done}"
    )
    assert done is True
    assert info["goal_reached"] is True

    # 7. Enhanced Campus Special Zones (Phase 15)
    print("\n[TEST 7] Testing Special Zones & Configurable Rewards (Phase 15):")
    # A) Safe Zone: (3, 2) from (3, 1) moving RIGHT (closer)
    env.agent_pos = (3, 1)
    next_s, reward, done, info = env.step(config.ACTION_RIGHT)
    print(f"  Step into Safe Zone (3, 2)   -> State: {next_s} | Reward: {reward:+5.1f} | Event: {info['event']}")
    assert next_s == (3, 2)
    assert reward == 7.0  # +5.0 (closer) + 2.0 (safe bonus)
    assert info["is_safe_zone"] is True
    assert info["event"] == "entered_safe_zone"

    # B) Crowded Zone: (6, 3) from (6, 2) moving RIGHT (closer)
    env.agent_pos = (6, 2)
    next_s, reward, done, info = env.step(config.ACTION_RIGHT)
    print(f"  Step into Crowded Zone (6, 3)-> State: {next_s} | Reward: {reward:+5.1f} | Event: {info['event']}")
    assert next_s == (6, 3)
    assert reward == 3.0  # +5.0 (closer) - 2.0 (crowd delay)
    assert info["is_crowded_zone"] is True
    assert info["event"] == "entered_crowded_zone"

    # C) Unsafe Zone: (5, 6) from (5, 5) moving RIGHT
    env.agent_pos = (5, 5)
    next_s, reward, done, info = env.step(config.ACTION_RIGHT)
    print(f"  Step into Unsafe Zone (5, 6) -> State: {next_s} | Reward: {reward:+5.1f} | Event: {info['event']}")
    assert next_s == (5, 6)
    assert reward == -10.0  # Flat unsafe hazard penalty
    assert info["is_unsafe_zone"] is True
    assert info["event"] == "entered_unsafe_zone"

    # D) Configurable properties verification
    print("  Testing Configurable Properties in CampusEnvironment:")
    custom_env = CampusEnvironment(
        reward_safe_zone=5.0,
        reward_crowded_zone=-4.0,
        reward_unsafe_zone=-25.0,
    )
    custom_env.agent_pos = (3, 1)
    _, rew_safe, _, _ = custom_env.step(config.ACTION_RIGHT)
    assert rew_safe == 10.0  # +5.0 (closer) + 5.0 (custom safe bonus)

    custom_env.agent_pos = (6, 2)
    _, rew_crowd, _, _ = custom_env.step(config.ACTION_RIGHT)
    assert rew_crowd == 1.0  # +5.0 (closer) - 4.0 (custom crowd delay)

    custom_env.agent_pos = (5, 5)
    _, rew_unsafe, _, _ = custom_env.step(config.ACTION_RIGHT)
    assert rew_unsafe == -25.0  # Custom unsafe penalty
    print(f"    [+] Custom zone rewards successfully applied: Safe={rew_safe:+.1f}, Crowd={rew_crowd:+.1f}, Hazard={rew_unsafe:+.1f}")

    print("\n======================================================================")
    print("               ALL PHASE 1 & 15 ENVIRONMENT TESTS PASSED!             ")
    print("======================================================================")


if __name__ == "__main__":
    test_campus_environment()
