"""
Mutation Engine: Orchestrates multi-round Evol-Instruct chains and tracks lineage trees.
"""

from typing import List, Dict, Any, Optional
from enum import Enum
import random

from .in_depth import add_constraint, deepen_reasoning, concretize_scenario
from .in_breadth import mutate_in_breadth


class MutationStrategy(str, Enum):
    ADD_CONSTRAINT = "add_constraint"
    DEEPEN_REASONING = "deepen_reasoning"
    CONCRETIZE = "concretize"
    IN_BREADTH = "in_breadth"


class MutationEngine:
    """
    Evol-Instruct generator that iteratively evolves seed instructions into complex prompts.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def evolve_step(
        self,
        instruction: str,
        strategy: Optional[MutationStrategy] = None,
    ) -> Dict[str, Any]:
        """Applies a single evolution mutation step to an instruction."""
        strat = strategy or random.choice(list(MutationStrategy))

        if strat == MutationStrategy.ADD_CONSTRAINT:
            evolved = add_constraint(instruction)
        elif strat == MutationStrategy.DEEPEN_REASONING:
            evolved = deepen_reasoning(instruction)
        elif strat == MutationStrategy.CONCRETIZE:
            evolved = concretize_scenario(instruction)
        elif strat == MutationStrategy.IN_BREADTH:
            evolved = mutate_in_breadth(instruction)
        else:
            raise ValueError(f"Unknown mutation strategy {strat}")

        return {
            "original_instruction": instruction,
            "evolved_instruction": evolved,
            "strategy": strat.value,
        }

    def evolve_chain(
        self,
        seed_instruction: str,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """
        Executes a multi-round evolution chain, building an evolution lineage trace.
        """
        if depth <= 0:
            raise ValueError(f"Evolution depth must be >= 1, got {depth}")

        current = seed_instruction
        history = []

        for step_idx in range(depth):
            step_res = self.evolve_step(current)
            history.append({
                "step": step_idx + 1,
                "strategy": step_res["strategy"],
                "instruction": step_res["evolved_instruction"],
            })
            current = step_res["evolved_instruction"]

        return {
            "seed_instruction": seed_instruction,
            "final_instruction": current,
            "depth": depth,
            "history": history,
        }
