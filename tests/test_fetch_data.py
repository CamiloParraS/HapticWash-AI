import hashlib

import pytest

from scripts import fetch_data


class _FakeResponse:
    def __init__(self, body: bytes):
        self.body = body
        self.headers = {"content-length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, _size):
        yield self.body


def test_fetch_verifies_and_skips(tmp_path, monkeypatch):
    body = b"sensor data"
    good = hashlib.md5(body).hexdigest()
    monkeypatch.setattr(fetch_data.requests, "get", lambda *a, **k: _FakeResponse(body))
    dest = tmp_path / "sub" / "a.csv"

    assert fetch_data.fetch("u", dest, good) is True
    assert dest.read_bytes() == body
    assert fetch_data.fetch("u", dest, good) is False  # already there -> skipped

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        fetch_data.fetch("u", tmp_path / "b.csv", "0" * 32)
    assert not list(tmp_path.glob("*.part"))  # no partial file left behind
