from src.transmitter.crc import encode_crc, verify_crc


def test_crc_roundtrip():
    codeword = encode_crc("1101", generator="1011")
    assert verify_crc(codeword, generator="1011") is True
