"""Very small educational AWGN/BPSK simulation helper."""
from __future__ import annotations

import math
import random

from src.utils.bit_utils import validate_bits


def bpsk_modulate(bits: str) -> list[float]:
    """Map 0 -> -1 and 1 -> +1."""
    validate_bits(bits)
    return [1.0 if bit == "1" else -1.0 for bit in bits]



def add_awgn(symbols: list[float], sigma: float, seed: int | None = None) -> list[float]:
    """Add Gaussian noise to BPSK symbols."""
    if sigma < 0:
        raise ValueError("Sigma must be non-negative.")
    rng = random.Random(seed)
    return [symbol + rng.gauss(0.0, sigma) for symbol in symbols]



def bpsk_demodulate(symbols: list[float]) -> str:
    """Hard-decision demodulation for BPSK."""
    return "".join("1" if value >= 0 else "0" for value in symbols)



def transmit_awgn(bits: str, sigma: float = 0.2, seed: int | None = None) -> str:
    """Full educational BPSK + AWGN + hard-decision pipeline."""
    validate_bits(bits)
    symbols = bpsk_modulate(bits)
    noisy_symbols = add_awgn(symbols, sigma=sigma, seed=seed)
    return bpsk_demodulate(noisy_symbols)
