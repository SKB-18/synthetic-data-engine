"""
Unit tests for QualityScorer clarity, complexity, and entropy scoring.
"""

import pytest
from curation.scorer import QualityScorer


def test_score_clarity_extremes():
    """Verify clarity scoring penalizes empty or very short instructions."""
    scorer = QualityScorer()
    assert scorer.score_clarity("") == 0.0
    assert scorer.score_clarity("hi") < 0.5

    # Well structured sentence
    good_text = "Explain the difference between TCP and UDP protocols in modern network programming."
    assert scorer.score_clarity(good_text) >= 0.8


def test_score_complexity():
    """Verify complexity scoring rewards technical terms and reasoning indicators."""
    scorer = QualityScorer()

    simple_text = "What is the weather like today in Seattle?"
    tech_text = (
        "Analyze the asymptotic trade-offs of distributed consensus algorithms like Raft, "
        "specifically evaluating latency, throughput, and invariant safety under network partitions."
    )

    score_simple = scorer.score_complexity(simple_text)
    score_tech = scorer.score_complexity(tech_text)

    assert score_tech > score_simple
    assert 0.0 <= score_tech <= 1.0


def test_score_entropy_repetitive_loop():
    """Verify entropy scoring strongly penalizes degenerate repetitive text."""
    scorer = QualityScorer()

    repetitive_text = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    diverse_text = "The quick brown fox jumps over the lazy dog and explores multiple invariants."

    ent_rep = scorer.score_entropy(repetitive_text)
    ent_div = scorer.score_entropy(diverse_text)

    assert ent_rep < 0.2
    assert ent_div > 0.8


def test_score_sample_composite():
    """Verify sample scoring returns all metrics and bounded composite score."""
    scorer = QualityScorer()
    inst = "Implement a balanced binary search tree with O(log N) lookup time."
    resp = "Here is the implementation of an AVL tree maintaining balance invariants."

    scores = scorer.score_sample(inst, response=resp)

    assert "clarity" in scores
    assert "complexity" in scores
    assert "entropy" in scores
    assert "overall_quality" in scores
    assert "response_quality" in scores

    assert 0.0 <= scores["overall_quality"] <= 1.0
    assert 0.0 <= scores["response_quality"] <= 1.0
