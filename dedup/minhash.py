"""
MinHash: Dimensionality reduction for Jaccard similarity estimation via random permutations.
"""

from typing import List, Set, Tuple, Optional, Any, Union
import random
import hashlib


# Large Mersenne prime 2^61 - 1
MERSENNE_PRIME = (1 << 61) - 1
MAX_HASH = (1 << 32) - 1


class MinHash:
    """
    Computes MinHash signatures over shingled text to estimate Jaccard similarity in O(K) time.
    """

    def __init__(self, num_perm: int = 128, seed: int = 42):
        if num_perm <= 0:
            raise ValueError(f"num_perm must be positive, got {num_perm}")
        self.num_perm = num_perm

        # Generate K random linear permutation hash coefficients: (a * x + b) % p
        rng = random.Random(seed)
        self.a_coeffs = [rng.randint(1, MAX_HASH) for _ in range(num_perm)]
        self.b_coeffs = [rng.randint(0, MAX_HASH) for _ in range(num_perm)]

    @staticmethod
    def get_shingles(text: str, k: int = 3) -> Set[str]:
        """Extracts character k-shingles from text."""
        clean_text = " ".join(text.lower().split())
        if len(clean_text) < k:
            return {clean_text} if clean_text else set()
        return {clean_text[i : i + k] for i in range(len(clean_text) - k + 1)}

    @staticmethod
    def _hash_token(token: str) -> int:
        """Hashes a string token into a 32-bit unsigned integer using MD5."""
        return int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16)

    def compute_signature(self, shingles: Any) -> List[int]:
        """
        Computes the K-element MinHash signature vector for a set of shingles (or raw text).
        """
        if isinstance(shingles, str):
            shingles = self.get_shingles(shingles)

        if not shingles:
            return [0] * self.num_perm

        token_hashes = [self._hash_token(s) for s in shingles]
        signature = [float("inf")] * self.num_perm

        for x in token_hashes:
            for i in range(self.num_perm):
                # Universal hash permutation
                val = ((self.a_coeffs[i] * x + self.b_coeffs[i]) % MERSENNE_PRIME) & MAX_HASH
                if val < signature[i]:
                    signature[i] = val

        return [int(val) for val in signature]

    def create_signature_from_text(self, text: str, k: int = 3) -> List[int]:
        """Convenience method: shingles text and computes signature."""
        shingles = self.get_shingles(text, k=k)
        return self.compute_signature(shingles)

    @staticmethod
    def estimate_jaccard(sig1: List[int], sig2: List[int]) -> float:
        """
        Estimates Jaccard similarity between two sets as the fraction of matching hash values.
        """
        if len(sig1) != len(sig2):
            raise ValueError(f"Signature length mismatch: {len(sig1)} vs {len(sig2)}")
        if not sig1:
            return 0.0

        matches = sum(1 for h1, h2 in zip(sig1, sig2) if h1 == h2)
        return matches / float(len(sig1))

    @staticmethod
    def exact_jaccard(set1: Set[str], set2: Set[str]) -> float:
        """Computes true exact Jaccard similarity: |A ∩ B| / |A ∪ B|."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / float(union)
