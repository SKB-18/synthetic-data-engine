"""
Unit tests for Evol-Instruct mutation operators and MutationEngine.
"""

import pytest
from mutation.in_depth import add_constraint, deepen_reasoning, concretize_scenario
from mutation.in_breadth import mutate_in_breadth
from mutation.engine import MutationEngine, MutationStrategy


def test_add_constraint():
    """Verify add_constraint appends an operational constraint while preserving the prompt."""
    prompt = "Write a binary search algorithm."
    evolved = add_constraint(prompt, constraint="Must run in O(log N) time.")
    assert prompt in evolved
    assert "Must run in O(log N) time." in evolved


def test_deepen_reasoning():
    """Verify deepen_reasoning appends analytical deduction requests."""
    prompt = "Explain quicksort."
    evolved = deepen_reasoning(prompt, prompt="Provide a formal proof of average case complexity.")
    assert prompt in evolved
    assert "formal proof" in evolved


def test_concretize_scenario():
    """Verify concretize_scenario injects domain scenario context."""
    prompt = "Implement a cache."
    evolved = concretize_scenario(prompt, scenario="Contextualize in an embedded device.")
    assert prompt in evolved
    assert "embedded device" in evolved


def test_mutate_in_breadth():
    """Verify mutate_in_breadth expands across domains."""
    prompt = "Explain tree traversal."
    evolved = mutate_in_breadth(prompt, domain="compiler design")
    assert prompt in evolved
    assert "compiler design" in evolved


def test_mutation_engine_step():
    """Verify single-step evolution returns strategy metadata."""
    engine = MutationEngine(seed=42)
    prompt = "Design a database index."
    res = engine.evolve_step(prompt, strategy=MutationStrategy.ADD_CONSTRAINT)

    assert res["original_instruction"] == prompt
    assert res["strategy"] == "add_constraint"
    assert len(res["evolved_instruction"]) > len(prompt)


def test_mutation_engine_chain():
    """Verify multi-round evolution trace depth and progressive lengthening."""
    engine = MutationEngine(seed=42)
    prompt = "Explain gradient descent."
    res = engine.evolve_chain(prompt, depth=3)

    assert res["seed_instruction"] == prompt
    assert res["depth"] == 3
    assert len(res["history"]) == 3
    assert len(res["final_instruction"]) > len(prompt)

    # Check each step in history
    for idx, step in enumerate(res["history"]):
        assert step["step"] == idx + 1
        assert "strategy" in step
        assert "instruction" in step


def test_mutation_engine_invalid_depth():
    """Verify ValueError is raised if evolution depth <= 0."""
    engine = MutationEngine()
    with pytest.raises(ValueError, match="depth must be >= 1"):
        engine.evolve_chain("Prompt", depth=0)
