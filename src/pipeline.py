"""End-to-end educational pipeline: message -> CRC -> Hamming -> channel -> decode -> CRC verify."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

from src.channel.awgn import transmit_awgn
from src.channel.bsc import inject_single_error, transmit_bsc
from src.receiver.decoder import decode_hamming_7_4
from src.transmitter.crc import encode_crc, verify_crc
from src.transmitter.hamming import encode_hamming_7_4
from src.utils.bit_utils import chunk_bits, validate_bits
from src.utils.metrics import calculate_ber


ChannelType = Literal["ideal", "manual", "bsc", "awgn"]


@dataclass
class PipelineResult:
    original_bits: str
    crc_generator: str
    crc_codeword: str
    padding_bits: int
    padded_crc_bits: str
    hamming_blocks_tx: list[str]
    transmitted_frame: str
    received_frame: str
    channel_type: str
    hamming_blocks_rx: list[str]
    syndromes: list[int]
    corrected_positions: list[int]
    corrected_blocks: list[str]
    decoded_padded_bits: str
    decoded_crc_bits: str
    recovered_bits: str
    crc_valid: bool
    ber_channel: float
    ber_end_to_end: float

    def to_dict(self) -> dict:
        return asdict(self)


def _pad_to_multiple_of_4(bits: str) -> tuple[str, int]:
    validate_bits(bits)
    padding = (-len(bits)) % 4
    return bits + ("0" * padding), padding


def _encode_hamming_blocks(bits: str) -> list[str]:
    return [encode_hamming_7_4(block) for block in chunk_bits(bits, 4)]


def _decode_hamming_blocks(frame: str) -> tuple[list[dict[str, str | int | bool]], str]:
    blocks = chunk_bits(frame, 7)
    if any(len(block) != 7 for block in blocks):
        raise ValueError("Received frame length must be a multiple of 7.")
    decoded = [decode_hamming_7_4(block) for block in blocks]
    padded_bits = "".join(str(item["data_bits"]) for item in decoded)
    return decoded, padded_bits


def run_pipeline(
    original_bits: str,
    generator: str = "1011",
    channel_type: ChannelType = "ideal",
    *,
    manual_error_position: int | None = None,
    bsc_p: float = 0.0,
    awgn_sigma: float = 0.2,
    seed: int | None = None,
) -> PipelineResult:
    """Run the full educational communication pipeline.

    Steps:
    1) Append CRC to original bits.
    2) Pad CRC-protected bits to a multiple of 4.
    3) Encode each 4-bit block with Hamming (7,4).
    4) Send through the chosen channel.
    5) Decode/correct each Hamming block.
    6) Remove padding and verify CRC.
    7) Recover original message if CRC frame is valid.
    """
    validate_bits(original_bits)

    crc_codeword = encode_crc(original_bits, generator=generator)
    padded_crc_bits, padding_bits = _pad_to_multiple_of_4(crc_codeword)
    tx_blocks = _encode_hamming_blocks(padded_crc_bits)
    transmitted_frame = "".join(tx_blocks)

    if channel_type == "ideal":
        received_frame = transmitted_frame
    elif channel_type == "manual":
        if manual_error_position is None:
            raise ValueError("manual_error_position is required when channel_type='manual'.")
        received_frame = inject_single_error(transmitted_frame, manual_error_position)
    elif channel_type == "bsc":
        received_frame = transmit_bsc(transmitted_frame, p=bsc_p, seed=seed)
    elif channel_type == "awgn":
        received_frame = transmit_awgn(transmitted_frame, sigma=awgn_sigma, seed=seed)
    else:
        raise ValueError(f"Unsupported channel type: {channel_type}")

    decoded_blocks, decoded_padded_bits = _decode_hamming_blocks(received_frame)
    decoded_crc_bits = decoded_padded_bits[:-padding_bits] if padding_bits else decoded_padded_bits
    crc_valid = verify_crc(decoded_crc_bits, generator=generator)
    recovered_bits = decoded_crc_bits[: -(len(generator) - 1)] if crc_valid else decoded_crc_bits[: -(len(generator) - 1)]

    return PipelineResult(
        original_bits=original_bits,
        crc_generator=generator,
        crc_codeword=crc_codeword,
        padding_bits=padding_bits,
        padded_crc_bits=padded_crc_bits,
        hamming_blocks_tx=tx_blocks,
        transmitted_frame=transmitted_frame,
        received_frame=received_frame,
        channel_type=channel_type,
        hamming_blocks_rx=[str(item["received"]) for item in decoded_blocks],
        syndromes=[int(item["syndrome"]) for item in decoded_blocks],
        corrected_positions=[int(item["syndrome"]) for item in decoded_blocks],
        corrected_blocks=[str(item["corrected"]) for item in decoded_blocks],
        decoded_padded_bits=decoded_padded_bits,
        decoded_crc_bits=decoded_crc_bits,
        recovered_bits=recovered_bits,
        crc_valid=crc_valid,
        ber_channel=calculate_ber(transmitted_frame, received_frame),
        ber_end_to_end=calculate_ber(original_bits, recovered_bits),
    )
