"""
Campus Navigator: RL Experimentation & Benchmarking Framework (Phase 20)
========================================================================
Module: experiments/__init__.py

Exports:
- ExperimentConfig: Dataclass specifying experimental hyperparameters.
- ExperimentResult: Dataclass holding measured empirical metrics.
- run_single_experiment: Runs training and evaluation for a single configuration.
- run_standard_experiments: Runs the 3 required benchmark experiments (500, 1000, 2000 eps).
- run_custom_experiments: Runs arbitrary configuration sets.
- plot_experiment_comparison: Generates 4-panel comparison bar charts.
- plot_learning_curves_comparison: Overlays convergence curves across experiments.
"""

from .framework import (
    ExperimentConfig,
    ExperimentResult,
    run_single_experiment,
    run_standard_experiments,
    run_custom_experiments,
    plot_experiment_comparison,
    plot_learning_curves_comparison,
    format_experiment_summary_table,
    print_experiment_summary_table,
)
from .reward_experiments import (
    RewardProfile,
    RewardExperimentResult,
    PROFILE_A,
    PROFILE_B,
    PROFILE_C,
    run_single_reward_experiment,
    run_reward_profile_experiments,
    plot_reward_profile_comparison,
    plot_reward_learning_curves,
    plot_reward_routes_comparison,
    format_reward_summary_table,
    print_reward_summary_table,
)

__all__ = [
    # Phase 20: Horizon Benchmarks
    "ExperimentConfig",
    "ExperimentResult",
    "run_single_experiment",
    "run_standard_experiments",
    "run_custom_experiments",
    "plot_experiment_comparison",
    "plot_learning_curves_comparison",
    "format_experiment_summary_table",
    "print_experiment_summary_table",
    # Phase 21: Reward Profiles
    "RewardProfile",
    "RewardExperimentResult",
    "PROFILE_A",
    "PROFILE_B",
    "PROFILE_C",
    "run_single_reward_experiment",
    "run_reward_profile_experiments",
    "plot_reward_profile_comparison",
    "plot_reward_learning_curves",
    "plot_reward_routes_comparison",
    "format_reward_summary_table",
    "print_reward_summary_table",
]
