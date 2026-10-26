"""
Stress tests, large-scale corpus indexing, and mathematical stability tests for synthetic-data-engine.
"""

import pytest
import time
from mutation.engine import MutationEngine
from curation.scorer import QualityScorer
from curation.filter import QualityFilter
from dedup.minhash import MinHash
from dedup.lsh import MinHashLSH
from preference.synthesizer import PreferenceSynthesizer


def test_lsh_large_corpus_scale():
    """Verify LSH scales smoothly across 500 documents with high throughput."""
    num_docs = 500
    minhash = MinHash(num_perm=64, seed=42)
    lsh = MinHashLSH(num_perm=64, num_bands=8, rows_per_band=8)

    # Generate synthetic documents with embedded duplicate clusters
    docs = []
    for i in range(num_docs):
        cluster_id = i % 25
        docs.append({
            "id": f"doc_{i}",
            "instruction": f"Algorithm cluster {cluster_id}: Implement distributed lock and state machine replication invariant {cluster_id}.",
        })

    t0 = time.perf_counter()
    unique, stats = lsh.deduplicate_dataset(docs, minhash=minhash, text_key="instruction")
    elapsed = time.perf_counter() - t0

    # 500 documents with 25 distinct templates should reduce to ~25 unique items
    assert len(unique) <= 30
    assert stats["duplicates_removed"] >= 470
    assert elapsed < 3.0  # Must complete sub-quadratic processing in under 3s


def test_deep_evol_chain_stability():
    """Verify deep evolution (depth=8) retains structural integrity and monotonically escalates complexity."""
    engine = MutationEngine(seed=99)
    seed = "Implement binary search."
    scorer = QualityScorer()

    chain_res = engine.evolve_chain(seed, depth=8)
    history = chain_res["history"]

    assert len(history) == 8
    seed_complexity = scorer.score_complexity(seed)

    complexities = [seed_complexity]
    for step in history:
        assert len(step["instruction"]) > 0
        score = scorer.score_complexity(step["instruction"])
        complexities.append(score)

    final_instruction = chain_res["final_instruction"]
    final_complexity = scorer.score_complexity(final_instruction)

    # Complexity at depth 8 should be significantly higher than initial seed
    assert final_complexity > seed_complexity
    assert len(final_instruction) > len(seed)


def test_scorer_unicode_code_latex():
    """Verify scorer handles raw Python code snippets, mathematical LaTeX, and Unicode without errors."""
    scorer = QualityScorer()

    latex_code_prompt = (
        "Prove that $\\sum_{i=1}^n i = \\frac{n(n+1)}{2}$ using structural induction. "
        "Here is the reference snippet:\n"
        "```python\n"
        "def compute_sum(n: int) -> int:\n"
        "    return (n * (n + 1)) // 2\n"
        "```\n"
        "Ensure memory invariance across thread executions: 🚀 π ≈ 3.14159."
    )

    scores = scorer.score_sample(latex_code_prompt)

    assert 0.0 <= scores["clarity"] <= 1.0
    assert 0.0 <= scores["complexity"] <= 1.0
    assert 0.0 <= scores["entropy"] <= 1.0
    assert 0.0 <= scores["overall_quality"] <= 1.0
    assert scores["entropy"] > 0.6  # High diverse character entropy


def test_preference_synthesis_determinism():
    """Verify preference synthesizer produces bitwise identical results given the same seed."""
    prompt = "Explain raft consensus algorithm invariants."
    completion = (
        "Raft achieves consensus via leader election, log replication, and commitment safety. "
        "The leader never overwrites its own log entries; log matching ensures consistency."
    )

    synth1 = PreferenceSynthesizer(seed=12345)
    pair1 = synth1.synthesize_pair_from_completion(prompt, completion, strategy="truncation")

    synth2 = PreferenceSynthesizer(seed=12345)
    pair2 = synth2.synthesize_pair_from_completion(prompt, completion, strategy="truncation")

    assert pair1 == pair2
    assert pair1["reward_margin"] == pair2["reward_margin"]
    assert pair1["rejected"] == pair2["rejected"]
