"""
Unit tests for synthetic-data-engine CLI subcommands.
"""

import sys
import subprocess
import pytest


def run_cli(*args):
    """Executes cli.py via subprocess and returns CompletedProcess."""
    cmd = [sys.executable, "cli.py"] + list(args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
    )


def test_cli_evolve():
    """Verify evolve subcommand runs and outputs lineage trace."""
    res = run_cli("evolve", "--prompt", "Write a binary search algorithm.", "--depth", "2", "--seed", "42")
    assert res.returncode == 0
    assert "Evol-Instruct Mutation" in res.stdout
    assert "Evolution Lineage Trace" in res.stdout
    assert "[Step 1]" in res.stdout
    assert "[Step 2]" in res.stdout


def test_cli_score():
    """Verify score subcommand evaluates instruction and response."""
    res = run_cli(
        "score",
        "--prompt", "Design a lock-free concurrent queue in Rust with memory ordering invariants.",
        "--response", "A lock-free queue uses atomic pointers and compare-and-swap primitives to preserve safety."
    )
    assert res.returncode == 0
    assert "Synthetic Sample Quality Auditor" in res.stdout
    assert "Clarity Score:" in res.stdout
    assert "Complexity Score:" in res.stdout
    assert "Overall Quality:" in res.stdout
    assert "Response Quality:" in res.stdout


def test_cli_dedup():
    """Verify dedup subcommand processes samples and reports reduction ratio."""
    res = run_cli("dedup", "--threshold", "0.65")
    assert res.returncode == 0
    assert "MinHash LSH Fuzzy Deduplication" in res.stdout
    assert "Total Documents Input:" in res.stdout
    assert "Unique Documents Retained:" in res.stdout
    assert "Near-Duplicates Purged:" in res.stdout


def test_cli_preference():
    """Verify preference subcommand synthesizes DPO pair with positive margin."""
    res = run_cli("preference", "--prompt", "Explain how memory allocation works in C programs.")
    assert res.returncode == 0
    assert "DPO Preference Pair Synthesizer" in res.stdout
    assert "Prompt:" in res.stdout
    assert "Chosen:" in res.stdout
    assert "Rejected:" in res.stdout
    assert "Reward Margin: +" in res.stdout


def test_cli_benchmark():
    """Verify benchmark subcommand executes the empirical evaluation suite."""
    res = run_cli("benchmark")
    assert res.returncode == 0
    assert "MinHash LSH vs. O(N^2) All-Pairs Deduplication Benchmark" in res.stdout
    assert "MinHash Jaccard Estimation Error Analysis" in res.stdout
    assert "Evol-Instruct Complexity & Quality Shift Benchmark" in res.stdout
