"""
In-Depth Evolution: Deepens instruction complexity, adds constraints, and demands formal reasoning.
"""

from typing import List, Dict, Optional
import random


CONSTRAINTS = [
    "Ensure your solution achieves O(N) time complexity and O(1) auxiliary space complexity.",
    "Handle all edge cases including empty inputs, null pointers, and maximum integer overflows.",
    "Provide the implementation strictly using standard library primitives without external dependencies.",
    "Include a formal mathematical invariant proof explaining why the algorithm terminates correctly.",
    "Ensure your response is compliant with safety guidelines and production concurrency requirements.",
]

REASONING_PROMPTS = [
    "First, provide a first-principles derivation of why this approach works before detailing the solution.",
    "Analyze potential failure modes and trade-offs compared to naive alternatives.",
    "Detail step-by-step reasoning using formal logical deduction.",
    "Formulate a counterexample that demonstrates why simpler heuristics fail.",
]

CONCRETIZATIONS = [
    "Ground your explanation in a high-throughput financial trading system scenario.",
    "Contextualize this in an embedded device with strict 64KB RAM memory constraints.",
    "Frame your explanation around a distributed multi-region database cluster encountering network partitions.",
    "Apply this to an autonomous robotics real-time control loop operating at 100Hz.",
]


def add_constraint(instruction: str, constraint: Optional[str] = None) -> str:
    """Evolves the instruction by appending an operational or algorithmic constraint."""
    chosen_constraint = constraint or random.choice(CONSTRAINTS)
    return f"{instruction.strip()} Operational constraint: {chosen_constraint}"


def deepen_reasoning(instruction: str, prompt: Optional[str] = None) -> str:
    """Evolves the instruction by demanding formal logical derivation or trade-off analysis."""
    chosen_prompt = prompt or random.choice(REASONING_PROMPTS)
    return f"{instruction.strip()} {chosen_prompt}"


def concretize_scenario(instruction: str, scenario: Optional[str] = None) -> str:
    """Evolves the instruction by grounding an abstract prompt in a realistic concrete scenario."""
    chosen_scenario = scenario or random.choice(CONCRETIZATIONS)
    return f"{instruction.strip()} {chosen_scenario}"
