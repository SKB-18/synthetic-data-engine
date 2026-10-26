"""
Deduplication module using MinHash and Locality-Sensitive Hashing (LSH).
"""

from .minhash import MinHash
from .lsh import MinHashLSH

__all__ = ["MinHash", "MinHashLSH"]
