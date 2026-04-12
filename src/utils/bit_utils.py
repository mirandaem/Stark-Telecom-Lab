"""Utility functions for bit-string manipulation."""
from __future__ import annotations


def validate_bits(bits: str) -> str:
    """Validate that a string contains only binary digits."""
    if not bits:
        raise ValueError("The bit string cannot be empty.")
    if any(bit not in {"0", "1"} for bit in bits):
        raise ValueError("The bit string must contain only '0' and '1'.")
    return bits


def xor_bits(a: str, b: str) -> str:
    """Return the XOR of two bit strings of equal length."""
    validate_bits(a)
    validate_bits(b)
    if len(a) != len(b):
        raise ValueError("Bit strings must have the same length.")
    return "".join("1" if x != y else "0" for x, y in zip(a, b))



def flip_bit(bits: str, position: int) -> str:
    """Flip a bit using a 1-based position."""
    validate_bits(bits)
    if position < 1 or position > len(bits):
        raise ValueError("Position out of range.")
    index = position - 1
    flipped = "1" if bits[index] == "0" else "0"
    return bits[:index] + flipped + bits[index + 1 :]



def chunk_bits(bits: str, size: int) -> list[str]:
    """Split a bit string into fixed-size chunks."""
    validate_bits(bits)
    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    return [bits[i : i + size] for i in range(0, len(bits), size)]
