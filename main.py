"""
Campus Navigator: An Intelligent Path-Finding Game Using Q-Learning
===================================================================
Main Entry Point (main.py)

College Experiential-Learning Project:
"Develop a Simple Game using RL Concepts"

This file provides the primary launch interface for the project.
Modes available:
    1. Default (No arguments): Launches the interactive Pygame game & menu.
    2. Headless Training:   python main.py --train [--episodes 600]
    3. Headless Evaluation: python main.py --eval
    4. Generate Plots:      python main.py --plot
"""

import argparse
import os
import sys

from training.train import train_agent, train_multi_destinations
from training.evaluate import evaluate_agent, evaluate_policy, evaluate_multi_destinations
from visualization.plots import generate_all_plots, plot_comprehensive_dashboard


def ensure_project_directories() -> None:
    """
    Ensure all required project directories and baseline configuration exist
    before application or experiment subsystems execute.
    """
    import config

    directories = [
        getattr(config, "MODELS_DIR", "models"),
        getattr(config, "RESULTS_DIR", "results"),
        getattr(config, "PLOTS_DIR", os.path.join(getattr(config, "RESULTS_DIR", "results"), "plots")),
        getattr(config, "LOGS_DIR", "logs"),
        getattr(config, "ASSETS_DIR", "assets"),
        getattr(config, "DOCS_DIR", "docs"),
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Ensure config.json is populated with valid defaults if missing
    try:
        config.load_config()
    except Exception:
        pass


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for alternative execution modes."""
    parser = argparse.ArgumentParser(
        description="Campus Navigator: An Intelligent Path-Finding Game Using Q-Learning"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run headless Q-learning training in the terminal without opening the game window.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run policy evaluation on the currently saved Q-table.",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Train or evaluate across all 5 campus destinations (Phase 16).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save training performance charts (rewards, steps, epsilon) to results/ directory.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=600,
        help="Number of training episodes (default: 600).",
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Run Phase 20 RL Experiments (500, 1000, 2000 episodes) and generate comparison plots.",
    )
    parser.add_argument(
        "--reward-exp",
        action="store_true",
        help="Run Phase 21 Reward Function Experiments (Profiles A, B, C) and generate comparison plots.",
    )
    parser.add_argument(
        "--report",
        "--final-eval",
        dest="final_eval",
        action="store_true",
        help="Run Phase 22 Final Evaluation, generate final_report.json and final_results.csv.",
    )
    return parser.parse_args()


def main() -> None:
    """Main application execution dispatcher."""
    ensure_project_directories()
    args = parse_arguments()

    try:
        from utils.logger import log_info, log_error, log_exception
        log_info(f"Campus Navigator session started with arguments: {sys.argv[1:]}")
    except ImportError:
        pass

    try:
        if args.train:
            if args.multi:
                print("[+] Starting Multi-Destination Goal-Conditioned Training across all 5 landmarks...")
                agent = train_multi_destinations(episodes_per_dest=args.episodes)
                print("[+] Multi-destination training completed and saved to models/q_table_multi.npy!")
            else:
                print(f"[+] Starting training for {args.episodes} episodes...")
                agent, metrics_df = train_agent(episodes=args.episodes)
                plot_comprehensive_dashboard(metrics_df)
                print("[+] Training completed and performance charts generated in results/ directory!")
            sys.exit(0)

        elif args.eval:
            if args.multi:
                print("[+] Evaluating Goal-Conditioned Policy across all 5 destinations...")
                evaluate_multi_destinations()
            else:
                print("[+] Evaluating learned policy...")
                evaluate_policy(eval_episodes=50)
            sys.exit(0)

        elif args.plot:
            print("[+] Generating visualization plots...")
            saved = generate_all_plots()
            if saved:
                print("[+] All plots successfully created in results/plots/ directory!")
            else:
                print("[!] Could not generate plots. Please ensure training has been executed first.")
            sys.exit(0)

        elif args.experiment:
            print("[+] Running Phase 20 RL Experiments suite (500, 1000, 2000 episodes)...")
            from experiments.framework import run_standard_experiments
            run_standard_experiments()
            sys.exit(0)

        elif args.reward_exp:
            print("[+] Running Phase 21 Reward Function Experiments (Profiles A, B, C)...")
            from experiments.reward_experiments import run_reward_profile_experiments
            run_reward_profile_experiments()
            sys.exit(0)

        elif args.final_eval:
            print("[+] Generating Phase 22 Final Evaluation Report...")
            from training.final_evaluation import generate_final_evaluation_report
            generate_final_evaluation_report()
            sys.exit(0)

        # Default: launch interactive Pygame GUI
        try:
            from game.game import CampusNavigatorGame
        except ImportError as e:
            print("\n[!] Note: Pygame is required to render the 2D graphical game window.")
            print("[*] Install dependencies with: pip install -r requirements.txt")
            print("[*] You can also run headless RL training right away: python main.py --train\n")
            sys.exit(1)

        print("[+] Launching Campus Navigator Game...")
        try:
            game = CampusNavigatorGame()
            game.run()
        except RuntimeError as e:
            print(f"\n[!] Graphical interface terminated: {e}")
            sys.exit(1)
        except Exception as e:
            try:
                from utils.logger import log_exception
                log_exception(e, "CampusNavigatorGame.run (Interactive Session)")
            except ImportError:
                pass
            print(f"\n[!] Unexpected application error: {e}")
            print("[*] Diagnostic error details have been recorded in 'logs/campus_navigator.log'.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n[!] Operation cancelled by user (KeyboardInterrupt). Exiting cleanly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
