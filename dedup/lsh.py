"""
Locality-Sensitive Hashing (LSH) for sub-quadratic fuzzy near-duplicate clustering.
"""

from typing import List, Dict, Set, Tuple, Any, Optional
from collections import defaultdict
from .minhash import MinHash


class MinHashLSH:
    """
    Locality-Sensitive Hashing index for fast near-neighbor search over MinHash signatures.
    Replaces O(N^2) pairwise comparisons with sub-quadratic O(N) hash bucket lookups.
    """

    def __init__(
        self,
        num_perm: int = 128,
        num_bands: int = 16,
        seed: int = 42,
        rows_per_band: Optional[int] = None,
    ):
        if rows_per_band is not None:
            if num_bands * rows_per_band != num_perm:
                raise ValueError(
                    f"num_perm ({num_perm}) must equal num_bands ({num_bands}) * rows_per_band ({rows_per_band})"
                )
            self.rows_per_band = rows_per_band
        else:
            if num_perm % num_bands != 0:
                raise ValueError(f"num_perm ({num_perm}) must be divisible by num_bands ({num_bands})")
            self.rows_per_band = num_perm // num_bands

        self.num_perm = num_perm
        self.num_bands = num_bands
        self.minhash = MinHash(num_perm=num_perm, seed=seed)

        # Hash tables per band: band_idx -> bucket_key -> set of doc_ids
        self.band_tables: List[Dict[Tuple[int, ...], Set[str]]] = [
            defaultdict(set) for _ in range(num_bands)
        ]
        # Registry of stored signatures: doc_id -> signature
        self.signatures: Dict[str, List[int]] = {}

    @property
    def estimated_threshold(self) -> float:
        """S-curve 50% collision probability threshold: tau ~= (1/b)^(1/r)."""
        return (1.0 / self.num_bands) ** (1.0 / self.rows_per_band)

    @property
    def threshold(self) -> float:
        """Alias for estimated_threshold."""
        return self.estimated_threshold

    def insert(self, doc_id: str, signature: List[int]):
        """Inserts a document signature into the LSH index."""
        if len(signature) != self.num_perm:
            raise ValueError(f"Expected signature of length {self.num_perm}, got {len(signature)}")

        if doc_id in self.signatures:
            return
        self.signatures[doc_id] = signature

        for b in range(self.num_bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            band_key = tuple(signature[start:end])
            self.band_tables[b][band_key].add(doc_id)

    def query(self, signature: List[int], min_jaccard: Optional[float] = None) -> Set[str]:
        """
        Finds candidate matching document IDs sharing at least one band bucket.
        Optionally filters candidates by estimated MinHash Jaccard similarity.
        """
        if len(signature) != self.num_perm:
            raise ValueError(f"Expected signature of length {self.num_perm}, got {len(signature)}")
        candidates = set()
        for b in range(self.num_bands):
            start = b * self.rows_per_band
            end = start + self.rows_per_band
            band_key = tuple(signature[start:end])
            if band_key in self.band_tables[b]:
                candidates.update(self.band_tables[b][band_key])

        if min_jaccard is None:
            return candidates

        # Filter candidates by full signature Jaccard estimation
        verified = set()
        for cand_id in candidates:
            cand_sig = self.signatures[cand_id]
            sim = MinHash.estimate_jaccard(signature, cand_sig)
            if sim >= min_jaccard:
                verified.add(cand_id)

        return verified

    def deduplicate_dataset(
        self,
        samples: List[Dict[str, Any]],
        text_key: str = "instruction",
        similarity_threshold: Optional[float] = None,
        minhash: Optional[MinHash] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Deduplicates a dataset using online LSH indexing.
        Preserves first-seen canonical document and drops fuzzy duplicates.
        """
        threshold = similarity_threshold or self.estimated_threshold
        unique_samples = []
        duplicate_count = 0
        duplicate_pairs = []
        mh = minhash or self.minhash

        for idx, sample in enumerate(samples):
            doc_id = str(sample.get("id", idx))
            text = sample.get(text_key, "")
            sig = mh.create_signature_from_text(text)

            matches = self.query(sig, min_jaccard=threshold)
            if matches:
                # Duplicate detected!
                duplicate_count += 1
                matched_doc = next(iter(matches))
                duplicate_pairs.append({
                    "duplicate_id": doc_id,
                    "matched_canonical_id": matched_doc,
                })
            else:
                self.insert(doc_id, sig)
                unique_samples.append(sample)

        total = len(samples)
        stats = {
            "total_documents": total,
            "unique_retained": len(unique_samples),
            "duplicates_removed": duplicate_count,
            "dedup_ratio_pct": round((duplicate_count / total * 100.0) if total > 0 else 0.0, 2),
            "similarity_threshold": round(threshold, 3),
            "duplicate_pairs": duplicate_pairs,
            "clusters": duplicate_pairs,
        }

        return unique_samples, stats
