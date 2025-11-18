"""
Training module for MARL algorithms.

Provides Trainer and Evaluator classes for orchestrating
training loops and evaluation.
"""

from src.training.trainer import Trainer
from src.training.evaluator import Evaluator

__all__ = ['Trainer', 'Evaluator']
