"""Receiver-side verifiers."""
from __future__ import annotations

from src.transmitter.crc import verify_crc


def verify_received_crc(bits: str, generator: str = "1011") -> bool:
    """Verify CRC at the receiver."""
    return verify_crc(bits, generator=generator)
