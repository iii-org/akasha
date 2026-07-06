import pytest

from akasha.helper import encoding

pytestmark = pytest.mark.unit


def test_detect_encoding_reads_file_prefix(tmp_path):
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world", encoding="utf-8")

    detected = encoding.detect_encoding(sample)

    assert detected
    assert detected.lower().replace("-", "") in {"utf8", "ascii"}


def test_md5_and_mac_address_are_stable(monkeypatch):
    assert encoding.get_text_md5("akasha") == encoding.get_text_md5("akasha")

    monkeypatch.setattr(encoding.uuid, "getnode", lambda: 0x001122334455)
    mac_hash = encoding.get_mac_address()
    expected_raw = "".join(
        ["{:02x}".format((0x001122334455 >> elements) & 0xFF) for elements in range(0, 2 * 6, 2)][::-1]
    )

    assert len(mac_hash) == 32
    assert mac_hash == encoding.get_text_md5(expected_raw)
