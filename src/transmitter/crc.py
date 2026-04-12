"""CRC encoding and verification helpers."""
from __future__ import annotations

from src.utils.bit_utils import validate_bits


def mod2_division(dividend: str, divisor: str) -> str:
    """Perform modulo-2 division and return the remainder."""
    validate_bits(dividend)
    validate_bits(divisor)
    if divisor[0] != "1":
        raise ValueError("The divisor must start with 1.")

    dividend_list = list(dividend)
    divisor_len = len(divisor)

    for i in range(len(dividend) - divisor_len + 1):
        if dividend_list[i] == "1":
            for j in range(divisor_len):
                dividend_list[i + j] = "0" if dividend_list[i + j] == divisor[j] else "1"

    return "".join(dividend_list[-(divisor_len - 1) :])



def encode_crc(data_bits: str, generator: str = "1011") -> str:
    """Append CRC remainder to a data word."""
    validate_bits(data_bits)
    validate_bits(generator)
    padded = data_bits + "0" * (len(generator) - 1)
    remainder = mod2_division(padded, generator)
    return data_bits + remainder



def verify_crc(codeword: str, generator: str = "1011") -> bool:
    """Check whether a received codeword passes the CRC test."""
    validate_bits(codeword)
    validate_bits(generator)
    remainder = mod2_division(codeword, generator)
    return set(remainder) == {"0"} or remainder == "0" * len(remainder)
