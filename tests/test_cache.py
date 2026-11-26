import hashlib
from pathlib import Path

import pytest

from mosaic.cache import cache_dir


def test_default_cache_dir(monkeypatch):
    monkeypatch.delenv("MOSAIC_CACHE_DIR", raising=False)

    assert cache_dir() == Path("~/.cache/mosaic").expanduser()


def test_cache_dir_can_be_overridden(monkeypatch, tmp_path):
    root = tmp_path / "models"
    monkeypatch.setenv("MOSAIC_CACHE_DIR", str(root))

    assert cache_dir() == root


def test_msa_cache_follows_runtime_override(monkeypatch, tmp_path):
    """Check that every backend reads MSAs from the cache directory set at fetch time."""
    from mosaic.msa import fetch_a3m

    sequence = "ACDE"
    digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]

    for name in ("first", "second"):
        root = tmp_path / name
        monkeypatch.setenv("MOSAIC_CACHE_DIR", str(root))
        cached = root / "msa" / f"{digest}.a3m"
        cached.parent.mkdir(parents=True)
        cached.write_text(f">{name}\n{sequence}\n")

        assert fetch_a3m(sequence) == cached.read_text()


@pytest.mark.parametrize(
    "status", ["UNKNOWN", "RUNNING", "RATELIMIT", "NETWORK_ERROR", "COMPLETE"]
)
def test_msa_search_deadline(monkeypatch, status):
    from types import SimpleNamespace

    import httpx

    from mosaic import msa

    now = [0.0]
    monkeypatch.setattr(
        msa,
        "time",
        SimpleNamespace(
            monotonic=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s)
        ),
    )

    def request(client, method, url, *, timeout, **kwargs):
        assert 0 < timeout <= 3
        if status == "NETWORK_ERROR":
            raise httpx.ConnectError("offline")
        if url.startswith("/result/"):
            now[0] += 4
        return httpx.Response(
            200,
            json={"id": "job", "status": status},
            request=httpx.Request(method, "https://example.org"),
        )

    monkeypatch.setattr(httpx.Client, "request", request)
    with pytest.raises(TimeoutError, match="deadline"):
        msa.search("AC", timeout=3)
    assert now[0] in (3, 4)


def test_msa_atomic_cache(monkeypatch, tmp_path):
    from mosaic import msa

    monkeypatch.setenv("MOSAIC_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(msa, "search", lambda seq: ">q\nAC\n")
    with monkeypatch.context() as patch:

        def fail(source, target):
            assert source.read_text() == ">q\nAC\n"
            raise OSError("simulated rename failure")

        patch.setattr(Path, "replace", fail)
        with pytest.raises(OSError, match="rename"):
            msa.fetch_a3m("AC")
    assert list((tmp_path / "msa").iterdir()) == []
    assert msa.fetch_a3m("AC") == ">q\nAC\n"
    monkeypatch.setattr(msa, "search", lambda seq: pytest.fail("cache miss"))
    assert msa.fetch_a3m("AC") == ">q\nAC\n"
