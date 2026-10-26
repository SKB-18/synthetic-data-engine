"""
Preference pair synthesis module for DPO post-training alignment.
"""

from .synthesizer import PreferenceSynthesizer, PerturbationStrategy

__all__ = ["PreferenceSynthesizer", "PerturbationStrategy"]
