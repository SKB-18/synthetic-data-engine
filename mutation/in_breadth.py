"""
In-Breadth Evolution: Expands instruction diversity across adjacent technical domains and use cases.
"""

from typing import List, Dict, Optional
import random

DOMAIN_SHIFTS = [
    ("algorithms", "distributed systems", "Adapt the core concept of this algorithm to a distributed consensus protocol."),
    ("web development", "kernel systems", "Translate the architectural principles of this pattern into an OS kernel module design."),
    ("machine learning", "information retrieval", "Apply the mathematical optimization principles here to vector indexing and nearest neighbor search."),
    ("cryptography", "networking", "Frame this security concept within zero-trust network packet inspection."),
]

BROAD_TOPIC_TEMPLATES = [
    "Explore a closely related concept in {domain} that shares the same underlying mathematical structure.",
    "Contrast this mechanism with standard industry paradigms in {domain}.",
    "Design an end-to-end architecture applying this idea to modern {domain} systems.",
]


def mutate_in_breadth(instruction: str, domain: Optional[str] = None) -> str:
    """Evolves the instruction by cross-pollinating with adjacent domains."""
    target_domain = domain or random.choice(["distributed systems", "operating systems", "compiler theory", "high-performance computing"])
    template = random.choice(BROAD_TOPIC_TEMPLATES)
    augmentation = template.format(domain=target_domain)
    return f"{instruction.strip()} Furthermore, {augmentation}"
