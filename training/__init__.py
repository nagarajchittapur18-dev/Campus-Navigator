"""
Campus Navigator: Training Package
=================================
Contains the training loop (train.py) and evaluation routines (evaluate.py).
"""

from training.train import train_agent
from training.evaluate import evaluate_agent, evaluate_policy
from training.final_evaluation import generate_final_evaluation_report

__all__ = ["train_agent", "evaluate_agent", "evaluate_policy", "generate_final_evaluation_report"]
