from src.channel.bsc import inject_single_error


def test_inject_single_error():
    assert inject_single_error("1011", 2) == "1111"
