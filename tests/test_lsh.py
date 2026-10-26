"""
Unit tests for MinHash LSH (Locality-Sensitive Hashing) indexing and deduplication.
"""

import pytest
from dedup.minhash import MinHash
from dedup.lsh import MinHashLSH


def test_lsh_bands_rows_validation():
    """Verify LSH enforces num_perm == num_bands * rows_per_band."""
    # 128 != 10 * 10
    with pytest.raises(ValueError, match="num_perm"):
        MinHashLSH(num_perm=128, num_bands=10, rows_per_band=10)

    # Valid: 128 == 16 * 8
    lsh = MinHashLSH(num_perm=128, num_bands=16, rows_per_band=8)
    assert lsh.num_bands == 16
    assert lsh.rows_per_band == 8
    assert round(lsh.threshold, 3) == 0.707


def test_lsh_identical_signatures_query():
    """Verify documents with identical text collide in all LSH bands."""
    minhash = MinHash(num_perm=64, seed=42)
    lsh = MinHashLSH(num_perm=64, num_bands=8, rows_per_band=8)

    text_a = "Implement an asynchronous distributed consensus protocol using Raft."
    sig_a = minhash.compute_signature(text_a)
    sig_b = minhash.compute_signature(text_a)

    lsh.insert("doc_a", sig_a)
    candidates = lsh.query(sig_b)

    assert "doc_a" in candidates


def test_lsh_disjoint_signatures_query():
    """Verify completely dissimilar documents do not collide in LSH buckets."""
    minhash = MinHash(num_perm=128, seed=42)
    lsh = MinHashLSH(num_perm=128, num_bands=16, rows_per_band=8)

    text_a = "Write a python script for quicksort algorithm."
    text_b = "How to bake a chocolate cake with strawberries."

    sig_a = minhash.compute_signature(text_a)
    sig_b = minhash.compute_signature(text_b)

    lsh.insert("algo_doc", sig_a)
    candidates = lsh.query(sig_b)

    assert "algo_doc" not in candidates


def test_lsh_deduplicate_dataset():
    """Verify deduplicate_dataset removes near-duplicate entries while preserving unique items."""
    minhash = MinHash(num_perm=64, seed=42)
    lsh = MinHashLSH(num_perm=64, num_bands=8, rows_per_band=8)

    dataset = [
        {"id": 1, "instruction": "Explain quantum teleportation and entanglement in physics."},
        {"id": 2, "instruction": "Explain quantum teleportation and entanglement in physics!"},  # Near duplicate
        {"id": 3, "instruction": "Implement Dijkstra shortest path algorithm in C++."},
        {"id": 4, "instruction": "Explain quantum teleportation and entanglement in physics."},  # Exact duplicate
        {"id": 5, "instruction": "How to optimize PyTorch memory using gradient checkpointing."},
    ]

    deduped, stats = lsh.deduplicate_dataset(dataset, minhash=minhash, text_key="instruction")

    assert len(deduped) == 3
    remaining_ids = {d["id"] for d in deduped}
    assert 1 in remaining_ids
    assert 3 in remaining_ids
    assert 5 in remaining_ids
    assert len(stats["clusters"]) >= 1


def test_lsh_signature_length_mismatch():
    """Verify querying or inserting with wrong signature dimension raises ValueError."""
    lsh = MinHashLSH(num_perm=64, num_bands=8, rows_per_band=8)
    invalid_sig = [1, 2, 3]  # Length 3 != 64

    with pytest.raises(ValueError, match="Expected signature of length"):
        lsh.insert("doc_bad", invalid_sig)

    with pytest.raises(ValueError, match="Expected signature of length"):
        lsh.query(invalid_sig)
