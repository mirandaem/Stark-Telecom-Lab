from src.receiver.decoder import decode_hamming_7_4
from src.transmitter.hamming import encode_hamming_7_4


def test_hamming_encode_decode_roundtrip():
    encoded = encode_hamming_7_4("1011")
    result = decode_hamming_7_4(encoded)
    assert result["data_bits"] == "1011"
    assert result["syndrome"] == 0


def test_hamming_corrects_single_error():
    result = decode_hamming_7_4("0110111")
    assert result["corrected"] == "0110011"
    assert result["data_bits"] == "1011"
