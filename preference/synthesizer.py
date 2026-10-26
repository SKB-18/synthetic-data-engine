"""
Preference Pair Synthesizer: Constructs chosen vs rejected pairs for DPO & RLHF alignment.
"""

from typing import List, Dict, Any, Tuple, Optional
import random
import re
try:
    from ..curation.scorer import QualityScorer
except (ImportError, ValueError):
    from curation.scorer import QualityScorer


class PerturbationStrategy:
    TRUNCATION = "truncation"
    REPETITION = "repetition"
    FORMAT_DEGRADATION = "format_degradation"
    LOGICAL_INVERSION = "logical_inversion"


class PreferenceSynthesizer:
    """
    Synthesizes preference pairs (chosen, rejected) from candidate completions or perturbation rules.
    """

    def __init__(self, seed: Optional[int] = None):
        self.scorer = QualityScorer()
        if seed is not None:
            random.seed(seed)

    def perturb_response(
        self,
        response: str,
        strategy: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Creates a degraded rejected response by applying a controlled perturbation strategy.
        Returns: (perturbed_text, strategy_used)
        """
        strat = strategy or random.choice([
            PerturbationStrategy.TRUNCATION,
            PerturbationStrategy.REPETITION,
            PerturbationStrategy.FORMAT_DEGRADATION,
            PerturbationStrategy.LOGICAL_INVERSION,
        ])

        words = response.split()
        if not words:
            return "No response provided.", "empty"

        if strat == PerturbationStrategy.TRUNCATION or strat == "truncation":
            # Cut response abruptly in mid-thought
            cut_idx = max(3, len(words) // 3)
            perturbed = " ".join(words[:cut_idx]) + " ... and therefore the result is"
            strat = PerturbationStrategy.TRUNCATION
        elif strat == PerturbationStrategy.REPETITION or strat == "repetition":
            # Inject degenerate repetition loop
            prefix = " ".join(words[: min(10, len(words))])
            repeated = " similarly and likewise identically " * 6
            perturbed = prefix + repeated
            strat = PerturbationStrategy.REPETITION
        elif strat in (PerturbationStrategy.FORMAT_DEGRADATION, "format_degrade", "format_degradation"):
            # Lowercase everything and strip formatting/punctuation/markdown symbols
            text = response.lower()
            for ch in ["#", "*", "-", "_", "`", ">", ".", ",", ";", ":"]:
                text = text.replace(ch, " ")
            text = re.sub(r"^\s*\d+\s*", "", text, flags=re.MULTILINE)
            perturbed = " ".join(text.split())
            strat = "format_degrade"
        elif strat == PerturbationStrategy.LOGICAL_INVERSION or strat == "logical_inversion":
            # Inject contradiction
            perturbed = f"However, notice that the entire premise is completely incorrect. {response}"
            strat = PerturbationStrategy.LOGICAL_INVERSION
        else:
            perturbed = response[: len(response) // 2]
            strat = "fallback"

        return perturbed, strat

    def synthesize_pair_from_completion(
        self,
        instruction: str,
        high_quality_response: str,
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synthesizes a (chosen, rejected) pair given a ground truth response.
        Enforces that chosen response strictly dominates the degraded rejected response.
        """
        rejected, strat = self.perturb_response(high_quality_response, strategy=strategy)

        chosen_scores = self.scorer.score_sample(instruction, high_quality_response)
        rejected_scores = self.scorer.score_sample(instruction, rejected)

        chosen_q = chosen_scores.get("response_quality", 0.8)
        raw_rejected_q = rejected_scores.get("response_quality", 0.4)

        # Degradation penalty applied to ensure negative sample is strictly sub-optimal
        penalty = {
            PerturbationStrategy.TRUNCATION: 0.25,
            PerturbationStrategy.REPETITION: 0.35,
            PerturbationStrategy.LOGICAL_INVERSION: 0.30,
            "format_degrade": 0.20,
        }.get(strat, 0.20)

        rejected_q = min(raw_rejected_q, max(0.05, chosen_q - penalty))
        margin = max(0.1, chosen_q - rejected_q)

        return {
            "prompt": instruction,
            "chosen": high_quality_response,
            "rejected": rejected,
            "chosen_score": round(chosen_q, 4),
            "rejected_score": round(rejected_q, 4),
            "reward_chosen": round(chosen_q, 4),
            "reward_rejected": round(rejected_q, 4),
            "reward_margin": round(margin, 4),
            "degradation_strategy": strat,
            "perturbation_strategy": strat,
        }

    def synthesize_dataset(
        self,
        seed_samples: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        Batch synthesizes DPO preference pairs from a list of {instruction, response} samples.
        """
        dataset = []
        for sample in seed_samples:
            inst = sample.get("instruction", "")
            resp = sample.get("response", "")
            if not inst or not resp:
                continue
            pair = self.synthesize_pair_from_completion(inst, resp)
            dataset.append(pair)
        return dataset
