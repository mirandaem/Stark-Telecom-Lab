from src.pipeline import run_pipeline


def test_pipeline_ideal_channel_round_trip():
    result = run_pipeline("1011", generator="1011", channel_type="ideal")
    assert result.crc_valid is True
    assert result.recovered_bits == "1011"
    assert result.ber_channel == 0.0
    assert result.ber_end_to_end == 0.0


def test_pipeline_manual_single_error_is_corrected():
    result = run_pipeline(
        "1011",
        generator="1011",
        channel_type="manual",
        manual_error_position=3,
    )
    assert result.crc_valid is True
    assert result.recovered_bits == "1011"
    assert result.ber_end_to_end == 0.0
