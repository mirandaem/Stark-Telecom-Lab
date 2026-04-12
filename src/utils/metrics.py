"""Metrics for digital communication simulations."""
from __future__ import annotations

from .bit_utils import validate_bits


def calculate_ber(original: str, received: str) -> float:
    """Compute the bit error rate between two bit strings."""
    validate_bits(original)
    validate_bits(received)
    if len(original) != len(received):
        raise ValueError("Bit strings must have the same length.")
    errors = sum(1 for a, b in zip(original, received) if a != b)
    return errors / len(original)
