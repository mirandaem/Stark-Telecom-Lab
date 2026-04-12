"""Binary symmetric channel model."""
from __future__ import annotations

import random

from src.utils.bit_utils import validate_bits, flip_bit


def transmit_bsc(bits: str, p: float = 0.0, seed: int | None = None) -> str:
    """Transmit bits through a binary symmetric channel."""
    validate_bits(bits)
    if not 0.0 <= p <= 1.0:
        raise ValueError("Probability p must be between 0 and 1.")

    rng = random.Random(seed)
    out = []
    for bit in bits:
        if rng.random() < p:
            out.append("1" if bit == "0" else "0")
        else:
            out.append(bit)
    return "".join(out)



def inject_single_error(bits: str, position: int) -> str:
    """Force a single bit error at a 1-based position."""
    return flip_bit(bits, position)
