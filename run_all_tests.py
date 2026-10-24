"""
Campus Navigator: Master Test Suite Runner (Phase 19: Complete Testing)
======================================================================
Script: run_all_tests.py

Discovers and runs all unit, integration, and regression test suites across:
1. ENVIRONMENT: reset(), valid movement, invalid movement, goal detection,
                obstacle detection, reward calculation, terminal state.
2. AGENT:       Q-table initialization, action selection, Q-value update,
                epsilon decay, save/load model.
3. TRAINING:    training starts, training finishes, results are generated.
4. EVALUATION:  trained model loads, evaluation does not modify Q-table,
                metrics are calculated.
5. GAME:        game starts, menu navigation, human controls, AI mode,
                missing model handling.
6. RESILIENCE:  all 10 failure scenarios, corrupted models, logging.
7. MULTI-DEST:  goal-conditioned RL, 5 destinations, route modal.
8. UI/GRAPHICS: coordinate rulers, pulsing status badge, waypoint numbers.
"""

import os
import sys
import time
import unittest

# Ensure headless execution for Pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def run_master_test_suite() -> bool:
    """Run all test modules in tests/ and print formatted academic report."""
    print("\n" + "=" * 78)
    print("      CAMPUS NAVIGATOR: COMPLETE AUTOMATED TEST SUITE (PHASE 19)      ")
    print("=" * 78)

    loader = unittest.TestLoader()
    tests_dir = os.path.join(PROJECT_ROOT, "tests")

    # Categories required by Phase 19 specification
    suite_categories = [
        ("ENVIRONMENT", "test_environment.py", "Campus gridworld physics, moves, obstacles, rewards"),
        ("AGENT",       "test_agent.py",       "Tabular Q-Learning, Bellman updates, epsilon-greedy"),
        ("TRAINING",    "test_training.py",    "Training lifecycle, episode horizons, metrics generation"),
        ("EVALUATION",  "test_evaluation.py",  "Zero-epsilon policy evaluation, frozen Q-table weights"),
        ("GAME",        "test_game.py",        "Main menu, human WASD, Watch AI mode, missing model alert"),
        ("RESILIENCE",  "test_phase18_error_handling.py", "All 10 failure scenarios, corrupted weights, logging"),
        ("MULTI-DEST",  "test_multi_destinations.py",     "5 predefined landmarks, goal-conditioned tensor"),
        ("UI/GRAPHICS", "test_phase17_ui.py",  "Coordinate rulers, status pills, ripple animations"),
        ("EXPERIMENTS", "test_experiments.py", "500/1000/2000 ep benchmarks, metric calculation, CSV & plots"),
        ("REWARD-EXP",  "test_reward_experiments.py", "Profiles A/B/C, reward engineering, safety metrics, CSV & plots"),
        ("FINAL-EVAL",  "test_final_evaluation.py", "Phase 22 final report JSON/CSV, actual metrics verification"),
        ("WEB-APP",     "test_web_app.py",          "Gradio web interface, session state, SVG & D-pad"),
    ]

    total_tests = 0
    total_failures = 0
    total_errors = 0
    category_results = []

    start_time = time.time()

    for category, filename, description in suite_categories:
        filepath = os.path.join(tests_dir, filename)
        if not os.path.exists(filepath):
            continue

        module_name = f"tests.{filename[:-3]}"
        suite = loader.loadTestsFromName(module_name)
        runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w"))
        result = runner.run(suite)

        cat_tests = result.testsRun
        cat_fails = len(result.failures)
        cat_errs = len(result.errors)
        passed = (cat_fails == 0 and cat_errs == 0)

        total_tests += cat_tests
        total_failures += cat_fails
        total_errors += cat_errs

        status_str = "PASS" if passed else "FAIL"
        category_results.append((category, filename, cat_tests, cat_fails, cat_errs, status_str, description))

    elapsed = time.time() - start_time

    # Print Summary Table
    print(f"\n{'CATEGORY':<14} | {'MODULE':<30} | {'TESTS':<6} | {'STATUS':<6} | {'DESCRIPTION'}")
    print("-" * 78)

    for cat, mod, tests_cnt, fails_cnt, errs_cnt, status, desc in category_results:
        symbol = "[+]" if status == "PASS" else "[X]"
        print(f"{symbol} {cat:<10} | {mod:<30} | {tests_cnt:<6} | {status:<6} | {desc}")

    print("=" * 78)
    print(f" TOTAL TESTS EXECUTED : {total_tests}")
    print(f" TOTAL FAILURES       : {total_failures}")
    print(f" TOTAL ERRORS         : {total_errors}")
    print(f" TOTAL RUNTIME        : {elapsed:.3f} seconds")
    print(f" OVERALL OUTCOME      : {'ALL TESTS PASSED (100% SUCCESS)' if (total_failures == 0 and total_errors == 0) else 'TEST FAILURES DETECTED'}")
    print("=" * 78 + "\n")

    return (total_failures == 0 and total_errors == 0)


if __name__ == "__main__":
    success = run_master_test_suite()
    sys.exit(0 if success else 1)
