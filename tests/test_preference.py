"""
Unit tests for PreferenceSynthesizer and negative response perturbation strategies.
"""

import pytest
from preference.synthesizer import PreferenceSynthesizer


def test_perturb_truncation():
    """Verify truncation perturbation cuts response prematurely."""
    synth = PreferenceSynthesizer(seed=42)
    response = (
        "First, define the graph nodes and adjacency lists. "
        "Second, maintain a priority queue of distances. "
        "Third, iterate through unvisited neighbors and relax edges. "
        "Finally, return the shortest path vector."
    )

    perturbed, strat = synth.perturb_response(response, strategy="truncation")
    assert strat == "truncation"
    assert len(perturbed.split()) < len(response.split())
    assert "therefore the result is" in perturbed


def test_perturb_repetition():
    """Verify repetition perturbation creates degenerate token loops."""
    synth = PreferenceSynthesizer(seed=42)
    response = "To balance an AVL tree, check the balance factor and perform left or right rotations."

    perturbed, strat = synth.perturb_response(response, strategy="repetition")
    assert strat == "repetition"
    assert "similarly and likewise identically" in perturbed
    assert len(perturbed) > len(response)


def test_perturb_format_degradation():
    """Verify format degradation strips structure, numbering, and Markdown formatting."""
    synth = PreferenceSynthesizer(seed=42)
    response = (
        "### Key Principles\n"
        "1. Write clean unit tests\n"
        "2. Ensure deterministic seeds\n"
        "- High cohesion\n"
        "- Low coupling"
    )

    perturbed, strat = synth.perturb_response(response, strategy="format_degrade")
    assert strat == "format_degrade"
    assert "###" not in perturbed
    assert "1. " not in perturbed
    assert "- " not in perturbed
    assert perturbed.islower()


def test_perturb_logical_inversion():
    """Verify logical inversion alters truth conditions and causality."""
    synth = PreferenceSynthesizer(seed=42)
    response = "Quicksort has an average time complexity of O(n log n) because the partition divides elements."

    perturbed, strat = synth.perturb_response(response, strategy="logical_inversion")
    assert strat == "logical_inversion"
    assert "incorrect" in perturbed or "premise" in perturbed


def test_synthesize_pair_schema():
    """Verify synthesize_pair_from_completion generates valid DPO format with positive margin."""
    synth = PreferenceSynthesizer(seed=42)
    prompt = "Explain how copy-on-write operates in virtual memory."
    completion = (
        "Copy-on-write is a resource management optimization. Pages are initially shared "
        "between parent and child with read-only permissions. When either process attempts to write, "
        "a page fault is triggered, and the OS allocates a new private copy."
    )

    pair = synth.synthesize_pair_from_completion(prompt, completion, strategy="repetition")

    assert "prompt" in pair
    assert "chosen" in pair
    assert "rejected" in pair
    assert "reward_chosen" in pair
    assert "reward_rejected" in pair
    assert "reward_margin" in pair
    assert "perturbation_strategy" in pair

    assert pair["chosen"] == completion
    assert pair["prompt"] == prompt
    assert pair["reward_margin"] > 0
    assert pair["reward_chosen"] > pair["reward_rejected"]


def test_synthesize_dataset_batch():
    """Verify batch synthesis produces full pairwise dataset with verified constraints."""
    synth = PreferenceSynthesizer(seed=123)
    dataset = [
        {
            "instruction": "What is an idempotent API?",
            "response": "An idempotent API can be called multiple times without altering the final state beyond the initial call.",
        },
        {
            "instruction": "Define a deterministic finite automaton.",
            "response": "A DFA is a 5-tuple consisting of states, alphabet, transition function, start state, and accept states.",
        },
    ]

    pairs = synth.synthesize_dataset(dataset)
    assert len(pairs) == 2
    for p in pairs:
        assert p["reward_margin"] >= 0
        assert p["chosen"] != p["rejected"]
