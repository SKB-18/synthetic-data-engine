"""
Quality and Complexity Scorer: Multi-dimensional heuristic evaluation of synthetic samples.
"""

from typing import Dict, Any, Optional
import math
import re
from collections import Counter


class QualityScorer:
    """
    Evaluates synthetic instructions and responses on clarity, technical complexity, and diversity.
    """

    TECHNICAL_INDICATORS = {
        "complexity", "algorithm", "invariant", "constraint", "asymptotic",
        "concurrency", "distributed", "deterministic", "optimization", "amortized",
        "latency", "throughput", "memory", "recursion", "trade-off", "deduction",
        "proof", "counterexample", "first-principles", "robustness", "cache",
    }

    REASONING_CONNECTIVES = {
        "therefore", "because", "consequently", "furthermore", "specifically",
        "conversely", "specifically", "assuming", "provided", "moreover",
    }

    def score_clarity(self, text: str) -> float:
        """
        Scores readability and structural clarity in [0.0, 1.0].
        Penalizes extremely short text, lack of punctuation, and excessive whitespace.
        """
        words = text.split()
        if not words:
            return 0.0

        length = len(words)
        # Optimal length between 15 and 300 words
        if length < 5:
            len_score = 0.2
        elif length < 15:
            len_score = 0.6
        elif length <= 300:
            len_score = 1.0
        else:
            len_score = max(0.5, 1.0 - (length - 300) / 1000.0)

        # Sentence punctuation check
        has_punct = bool(re.search(r"[.?!:]", text))
        punct_score = 1.0 if has_punct else 0.4

        # Clean structure (capitalized start)
        cap_score = 1.0 if text and text[0].isupper() else 0.8

        return 0.5 * len_score + 0.3 * punct_score + 0.2 * cap_score

    def score_complexity(self, text: str) -> float:
        """
        Scores algorithmic and technical complexity in [0.0, 1.0].
        Evaluates presence of technical indicators and formal reasoning markers.
        """
        words = [w.lower().strip(".,;:!?()[]") for w in text.split()]
        if not words:
            return 0.0

        tech_count = sum(1 for w in words if w in self.TECHNICAL_INDICATORS)
        reason_count = sum(1 for w in words if w in self.REASONING_CONNECTIVES)

        tech_density = min(1.0, tech_count / 3.0)
        reason_density = min(1.0, reason_count / 2.0)
        vocab_richness = min(1.0, len(set(words)) / float(len(words)))

        return 0.5 * tech_density + 0.3 * reason_density + 0.2 * vocab_richness

    def score_entropy(self, text: str) -> float:
        """
        Computes Shannon entropy (combining character and word entropy) normalized to [0.0, 1.0].
        Penalizes repetitive degenerate patterns (e.g. 'aaaaa' or repeated word loops).
        """
        if not text:
            return 0.0

        counts = Counter(text)
        total = float(len(text))
        char_entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
        norm_char = min(1.0, char_entropy / 4.0)

        words = text.split()
        if len(words) > 1:
            w_counts = Counter(words)
            w_total = float(len(words))
            word_entropy = -sum((c / w_total) * math.log2(c / w_total) for c in w_counts.values())
            max_w = math.log2(w_total) if w_total > 1 else 1.0
            norm_word = word_entropy / max_w if max_w > 0 else 1.0
            return min(1.0, 0.5 * norm_char + 0.5 * norm_word)

        return norm_char

    def score_sample(self, instruction: str, response: Optional[str] = None) -> Dict[str, float]:
        """
        Computes comprehensive evaluation metrics for an instruction-response sample.
        """
        inst_clarity = self.score_clarity(instruction)
        inst_complexity = self.score_complexity(instruction)
        inst_entropy = self.score_entropy(instruction)

        # Base composite score
        overall = 0.4 * inst_clarity + 0.4 * inst_complexity + 0.2 * inst_entropy

        results = {
            "clarity": round(inst_clarity, 4),
            "complexity": round(inst_complexity, 4),
            "entropy": round(inst_entropy, 4),
            "overall_quality": round(overall, 4),
        }

        if response is not None:
            resp_clarity = self.score_clarity(response)
            resp_complexity = self.score_complexity(response)
            resp_entropy = self.score_entropy(response)
            results["response_clarity"] = round(resp_clarity, 4)
            results["response_complexity"] = round(resp_complexity, 4)
            results["response_entropy"] = round(resp_entropy, 4)
            results["response_quality"] = round(
                0.4 * resp_clarity + 0.4 * resp_complexity + 0.2 * resp_entropy, 4
            )

        return results
