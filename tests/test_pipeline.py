"""
End-to-end pipeline test for synthetic data generation, quality filtering, LSH deduplication, and DPO synthesis.
"""

import pytest
from mutation.engine import MutationEngine
from curation.scorer import QualityScorer
from curation.filter import QualityFilter
from dedup.minhash import MinHash
from dedup.lsh import MinHashLSH
from preference.synthesizer import PreferenceSynthesizer


def test_full_synthetic_data_lifecycle():
    """
    Test entire synthetic data generation pipeline:
    1. Seed prompt collection
    2. Evol-Instruct mutation (depth escalation)
    3. Heuristic quality and complexity filtering
    4. MinHash LSH near-duplicate elimination
    5. DPO preference pair synthesis
    """
    # 1. Seeds
    seeds = [
        "Implement a high-performance in-memory cache system.",
        "Explain binary search algorithm and its invariants.",
        "Write a function to reverse a singly linked list.",
        "How does HTTP caching work with conditional requests?",
    ]

    # 2. Evol-Instruct mutation
    mutator = MutationEngine(seed=42)
    evolved_dataset = []
    sample_id = 1

    for seed in seeds:
        chain_res = mutator.evolve_chain(seed, depth=2)
        evolved_dataset.append({
            "id": sample_id,
            "instruction": seed,
            "response": f"Base solution for {seed}",
        })
        sample_id += 1
        for step in chain_res["history"]:
            evolved_dataset.append({
                "id": sample_id,
                "instruction": step["instruction"],
                "response": f"Comprehensive solution and formal analysis for: {step['instruction']}. We establish the invariants and time complexity."
            })
            sample_id += 1

    assert len(evolved_dataset) == 4 * 3  # 4 seeds * (original + depth 1 + depth 2) = 12 samples

    # Add an explicit near-duplicate and a degenerate sample
    evolved_dataset.append({
        "id": 998,
        "instruction": evolved_dataset[0]["instruction"] + "!",  # Near duplicate
        "response": evolved_dataset[0]["response"],
    })
    evolved_dataset.append({
        "id": 999,
        "instruction": "bad bad bad bad bad bad bad bad bad bad bad bad",  # Degenerate
        "response": "bad response",
    })

    # 3. Quality Filtering
    qfilter = QualityFilter(min_overall=0.35, min_entropy=0.35, min_words=5)
    filtered_samples, filter_stats = qfilter.filter_dataset(evolved_dataset)

    assert filter_stats["rejected"] >= 1
    rejected_ids = {s["id"] for s in evolved_dataset} - {s["id"] for s in filtered_samples}
    assert 999 in rejected_ids  # Degenerate sample was eliminated

    # 4. MinHash LSH Deduplication
    minhash = MinHash(num_perm=64, seed=42)
    lsh = MinHashLSH(num_perm=64, num_bands=8, rows_per_band=8)

    deduped_samples, stats = lsh.deduplicate_dataset(
        filtered_samples, minhash=minhash, text_key="instruction"
    )

    assert len(deduped_samples) < len(filtered_samples)
    deduped_ids = {s["id"] for s in deduped_samples}
    # One of sample 1 or sample 998 should be removed
    assert not (1 in deduped_ids and 998 in deduped_ids)

    # 5. DPO Preference Pair Synthesis
    synth = PreferenceSynthesizer(seed=42)
    preference_pairs = synth.synthesize_dataset(deduped_samples)

    assert len(preference_pairs) == len(deduped_samples)
    for pair in preference_pairs:
        assert pair["reward_margin"] > 0
        assert pair["reward_chosen"] > pair["reward_rejected"]
        assert len(pair["chosen"]) > 0
        assert len(pair["rejected"]) > 0
