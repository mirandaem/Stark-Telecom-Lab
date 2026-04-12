"""Hamming (7,4) encoder for educational use."""
from __future__ import annotations

from src.utils.bit_utils import validate_bits


def encode_hamming_7_4(data_bits: str) -> str:
    """Encode a 4-bit word using the Hamming (7,4) code.

    Position layout: p1 p2 d1 p4 d2 d3 d4
    """
    validate_bits(data_bits)
    if len(data_bits) != 4:
        raise ValueError("Hamming (7,4) requires exactly 4 data bits.")

    d1, d2, d3, d4 = (int(bit) for bit in data_bits)
    p1 = (d1 + d2 + d4) % 2
    p2 = (d1 + d3 + d4) % 2
    p4 = (d2 + d3 + d4) % 2

    return f"{p1}{p2}{d1}{p4}{d2}{d3}{d4}"
