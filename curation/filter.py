"""
Quality Filter: Rejection gates for low-information, degenerate, or poorly structured samples.
"""

from typing import List, Dict, Any, Tuple, Optional
from .scorer import QualityScorer


class QualityFilter:
    """
    Filters datasets against quality, complexity, and entropy thresholds.
    """

    def __init__(
        self,
        min_overall: float = 0.45,
        min_clarity: float = 0.40,
        min_complexity: float = 0.15,
        min_entropy: float = 0.45,
        min_words: int = 5,
        max_words: int = 1000,
    ):
        self.min_overall = min_overall
        self.min_clarity = min_clarity
        self.min_complexity = min_complexity
        self.min_entropy = min_entropy
        self.min_words = min_words
        self.max_words = max_words
        self.scorer = QualityScorer()

    def evaluate_sample(self, sample: Dict[str, Any]) -> Tuple[bool, Optional[str], Dict[str, float]]:
        """
        Evaluates a single sample. Returns (is_accepted, rejection_reason, score_dict).
        """
        instruction = sample.get("instruction", "")
        words = instruction.split()

        if len(words) < self.min_words:
            return False, f"Too short ({len(words)} < {self.min_words} words)", {}
        if len(words) > self.max_words:
            return False, f"Too long ({len(words)} > {self.max_words} words)", {}

        response = sample.get("response")
        scores = self.scorer.score_sample(instruction, response=response)

        if scores["entropy"] < self.min_entropy:
            return False, f"Low entropy ({scores['entropy']} < {self.min_entropy})", scores
        if scores["clarity"] < self.min_clarity:
            return False, f"Low clarity ({scores['clarity']} < {self.min_clarity})", scores
        if scores["complexity"] < self.min_complexity:
            return False, f"Low complexity ({scores['complexity']} < {self.min_complexity})", scores
        if scores["overall_quality"] < self.min_overall:
            return False, f"Low overall quality ({scores['overall_quality']} < {self.min_overall})", scores

        return True, None, scores

    def filter_dataset(
        self,
        samples: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Filters a collection of samples and returns clean subset and rejection statistics.
        """
        passed = []
        rejections = {}

        for sample in samples:
            ok, reason, scores = self.evaluate_sample(sample)
            if ok:
                sample_with_scores = dict(sample)
                sample_with_scores["quality_metrics"] = scores
                passed.append(sample_with_scores)
            else:
                rejections[reason] = rejections.get(reason, 0) + 1

        total = len(samples)
        accepted = len(passed)
        acceptance_rate = (accepted / total * 100.0) if total > 0 else 0.0

        stats = {
            "total_evaluated": total,
            "passed": accepted,
            "rejected": total - accepted,
            "acceptance_rate_pct": round(acceptance_rate, 2),
            "rejection_breakdown": rejections,
        }

        return passed, stats
