"""Hamming (7,4) decoding and correction."""
from __future__ import annotations

from src.utils.bit_utils import flip_bit, validate_bits


def calculate_syndrome(codeword: str) -> int:
    """Return the 1-based error position from the syndrome."""
    validate_bits(codeword)
    if len(codeword) != 7:
        raise ValueError("Hamming (7,4) codewords must have length 7.")

    b = [int(bit) for bit in codeword]
    s1 = (b[0] + b[2] + b[4] + b[6]) % 2
    s2 = (b[1] + b[2] + b[5] + b[6]) % 2
    s4 = (b[3] + b[4] + b[5] + b[6]) % 2
    return s1 + 2 * s2 + 4 * s4



def decode_hamming_7_4(codeword: str) -> dict[str, str | int | bool]:
    """Decode and correct a Hamming (7,4) word."""
    validate_bits(codeword)
    if len(codeword) != 7:
        raise ValueError("Hamming (7,4) codewords must have length 7.")

    syndrome = calculate_syndrome(codeword)
    corrected = codeword
    corrected_error = False
    if syndrome != 0:
        corrected = flip_bit(codeword, syndrome)
        corrected_error = True

    data_bits = corrected[2] + corrected[4] + corrected[5] + corrected[6]
    return {
        "received": codeword,
        "syndrome": syndrome,
        "corrected": corrected,
        "corrected_error": corrected_error,
        "data_bits": data_bits,
    }
