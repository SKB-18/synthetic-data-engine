"""
Unit tests for QualityFilter gating and dataset curation.
"""

import pytest
from curation.filter import QualityFilter


def test_filter_length_gating():
    """Verify filter rejects samples that violate minimum word length."""
    qfilter = QualityFilter(min_words=5)

    too_short = {"instruction": "sort numbers"}
    valid = {"instruction": "Write a quicksort implementation handling duplicates."}

    ok_short, reason_short, _ = qfilter.evaluate_sample(too_short)
    ok_valid, reason_valid, _ = qfilter.evaluate_sample(valid)

    assert not ok_short
    assert "Too short" in reason_short
    assert ok_valid
    assert reason_valid is None


def test_filter_entropy_gating():
    """Verify filter rejects low-entropy repetitive prompts."""
    qfilter = QualityFilter(min_entropy=0.5)
    repetitive = {"instruction": "word word word word word word word word word"}

    ok, reason, _ = qfilter.evaluate_sample(repetitive)
    assert not ok
    assert "entropy" in reason.lower() or "quality" in reason.lower()


def test_filter_dataset_statistics():
    """Verify filter_dataset returns clean subset and correct statistical breakdown."""
    qfilter = QualityFilter(min_words=5, min_complexity=0.2)

    dataset = [
        {"id": 1, "instruction": "Tiny"},  # Too short
        {"id": 2, "instruction": "Explain the time complexity and invariants of binary search algorithms."},  # Passes
        {"id": 3, "instruction": "a a a a a a a a a a a a"},  # Low complexity / entropy
    ]

    passed, stats = qfilter.filter_dataset(dataset)

    assert len(passed) == 1
    assert passed[0]["id"] == 2
    assert stats["total_evaluated"] == 3
    assert stats["passed"] == 1
    assert stats["rejected"] == 2
    assert "quality_metrics" in passed[0]
