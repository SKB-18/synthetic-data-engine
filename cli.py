"""
Interactive and Scriptable CLI for Synthetic Data Generation, Quality Curation, and MinHash LSH Deduplication.
"""

import argparse
import sys
import os
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from mutation.engine import MutationEngine, MutationStrategy
from curation.scorer import QualityScorer
from curation.filter import QualityFilter
from dedup.minhash import MinHash
from dedup.lsh import MinHashLSH
from preference.synthesizer import PreferenceSynthesizer


def cmd_evolve(args):
    """Evolves a prompt using Evol-Instruct mutation strategies."""
    print("=" * 65)
    print(f" Evol-Instruct Mutation (Depth={args.depth})")
    print("=" * 65)
    print(f"  • Seed Instruction:  \"{args.prompt}\"")

    engine = MutationEngine(seed=args.seed)
    res = engine.evolve_chain(args.prompt, depth=args.depth)

    print("-" * 65)
    print(" Evolution Lineage Trace:")
    for step in res["history"]:
        print(f"   [Step {step['step']}] Strategy: {step['strategy']:<18}")
        print(f"          Result: \"{step['instruction']}\"")
    print("=" * 65)


def cmd_score(args):
    """Scores a text sample across clarity, complexity, and entropy."""
    print("=" * 65)
    print(" Synthetic Sample Quality Auditor")
    print("=" * 65)
    scorer = QualityScorer()
    scores = scorer.score_sample(args.prompt, response=args.response)

    print(f"  • Instruction:       \"{args.prompt}\"")
    print(f"  • Clarity Score:     {scores['clarity']:.4f}")
    print(f"  • Complexity Score:  {scores['complexity']:.4f}")
    print(f"  • Shannon Entropy:   {scores['entropy']:.4f}")
    print(f"  • Overall Quality:   {scores['overall_quality']:.4f}")

    if args.response:
        print(f"  • Response Quality:  {scores.get('response_quality', 0.0):.4f}")
    print("=" * 65)


def cmd_dedup(args):
    """Deduplicates a collection of instructions using MinHash LSH."""
    print("=" * 65)
    print(f" MinHash LSH Fuzzy Deduplication (Threshold={args.threshold})")
    print("=" * 65)

    samples = [
        {"id": "doc_1", "instruction": "Explain binary search implementation in Python with edge cases."},
        {"id": "doc_2", "instruction": "Explain binary search implementation in Python with edge cases and bounds."},  # Near-duplicate
        {"id": "doc_3", "instruction": "Describe Raft consensus leader election and log replication invariants."},
        {"id": "doc_4", "instruction": "Explain binary search implementation in Python with all edge cases."},  # Near-duplicate
        {"id": "doc_5", "instruction": "How do B-tree indexes speed up SQL range queries in relational databases?"},
    ]

    lsh = MinHashLSH(num_perm=128, num_bands=16)
    unique, stats = lsh.deduplicate_dataset(samples, similarity_threshold=args.threshold)

    print(f"  • Total Documents Input:      {stats['total_documents']}")
    print(f"  • Unique Documents Retained:  {stats['unique_retained']}")
    print(f"  • Near-Duplicates Purged:     {stats['duplicates_removed']} ({stats['dedup_ratio_pct']:.1f}% reduction)")
    print("-" * 65)
    print(" Retained Unique Documents:")
    for doc in unique:
        print(f"   [{doc['id']}] {doc['instruction']}")
    print("=" * 65)


def cmd_preference(args):
    """Synthesizes DPO preference pairs (chosen vs rejected) with reward margins."""
    print("=" * 65)
    print(" DPO Preference Pair Synthesizer")
    print("=" * 65)

    synth = PreferenceSynthesizer(seed=42)
    inst = args.prompt or "Explain how memory allocation works in C programs."
    resp = (
        "In C, dynamic memory allocation is managed through malloc, calloc, realloc, and free. "
        "These functions request memory from the operating system's heap. "
        "Each allocation must be explicitly freed to prevent memory leaks."
    )

    pair = synth.synthesize_pair_from_completion(inst, resp)

    print(f"  • Prompt:        \"{pair['prompt']}\"")
    print(f"  • Chosen:        \"{pair['chosen']}\" (Score: {pair['chosen_score']:.4f})")
    print(f"  • Rejected:      \"{pair['rejected']}\" (Score: {pair['rejected_score']:.4f})")
    print(f"  • Reward Margin: +{pair['reward_margin']:.4f} (Strategy: {pair['degradation_strategy']})")
    print("=" * 65)


def cmd_benchmark(args):
    """Runs empirical benchmark suite."""
    from benchmarks.run_benchmark import main as run_bench
    run_bench()


def main():
    parser = argparse.ArgumentParser(description="synthetic-data-engine CLI: Evol-Instruct, Quality Scoring, MinHash LSH, and Preference Synthesis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # evolve
    p_evol = subparsers.add_parser("evolve", help="Evolve an instruction via Evol-Instruct")
    p_evol.add_argument("--prompt", "-p", type=str, default="Write a binary search function.", help="Seed instruction")
    p_evol.add_argument("--depth", "-d", type=int, default=2, help="Evolution depth")
    p_evol.add_argument("--seed", type=int, default=42, help="Random seed")
    p_evol.set_defaults(func=cmd_evolve)

    # score
    p_score = subparsers.add_parser("score", help="Audit sample quality, complexity, and entropy")
    p_score.add_argument("--prompt", "-p", type=str, required=True, help="Instruction to score")
    p_score.add_argument("--response", "-r", type=str, default=None, help="Optional response to score")
    p_score.set_defaults(func=cmd_score)

    # dedup
    p_dedup = subparsers.add_parser("dedup", help="Deduplicate dataset using MinHash LSH")
    p_dedup.add_argument("--threshold", "-t", type=float, default=0.7, help="Jaccard similarity threshold")
    p_dedup.set_defaults(func=cmd_dedup)

    # preference
    p_pref = subparsers.add_parser("preference", help="Synthesize chosen vs rejected preference pair")
    p_pref.add_argument("--prompt", "-p", type=str, default=None, help="Instruction prompt")
    p_pref.set_defaults(func=cmd_preference)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run full benchmark suite")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
