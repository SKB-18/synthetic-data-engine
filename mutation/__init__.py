"""
Mutation module for Evol-Instruct synthetic data generation.
"""

from .in_depth import add_constraint, deepen_reasoning, concretize_scenario
from .in_breadth import mutate_in_breadth
from .engine import MutationEngine, MutationStrategy

__all__ = [
    "add_constraint",
    "deepen_reasoning",
    "concretize_scenario",
    "mutate_in_breadth",
    "MutationEngine",
    "MutationStrategy",
]
