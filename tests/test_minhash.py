"""
Unit tests for MinHash signature generation and Jaccard similarity estimation.
"""

import pytest
from dedup.minhash import MinHash


def test_minhash_shingles():
    """Verify character k-shingle extraction from text."""
    text = "hello"
    shingles = MinHash.get_shingles(text, k=3)
    # "hel", "ell", "llo"
    assert shingles == {"hel", "ell", "llo"}


def test_minhash_signature_length():
    """Verify computed signature matches specified permutation count."""
    minhash = MinHash(num_perm=64)
    shingles = {"token_a", "token_b", "token_c"}
    sig = minhash.compute_signature(shingles)
    assert len(sig) == 64
    assert all(isinstance(x, int) for x in sig)


def test_minhash_identical_documents():
    """Verify identical documents achieve estimated Jaccard similarity = 1.0."""
    minhash = MinHash(num_perm=128)
    text = "Explain the Raft consensus algorithm leader election."

    sig1 = minhash.create_signature_from_text(text)
    sig2 = minhash.create_signature_from_text(text)

    est_j = MinHash.estimate_jaccard(sig1, sig2)
    assert est_j == 1.0


def test_minhash_disjoint_documents():
    """Verify completely disjoint texts produce very low estimated Jaccard."""
    minhash = MinHash(num_perm=128)
    text1 = "11111111111111111111"
    text2 = "zzzzzzzzzzzzzzzzzzzz"

    sig1 = minhash.create_signature_from_text(text1)
    sig2 = minhash.create_signature_from_text(text2)

    est_j = MinHash.estimate_jaccard(sig1, sig2)
    assert est_j < 0.10


def test_exact_jaccard():
    """Verify exact set-theoretic Jaccard computation."""
    set_a = {"apple", "banana", "cherry"}
    set_b = {"banana", "cherry", "date"}

    # Intersection: {banana, cherry} = 2
    # Union: {apple, banana, cherry, date} = 4
    # Jaccard = 2/4 = 0.5
    assert MinHash.exact_jaccard(set_a, set_b) == 0.5
