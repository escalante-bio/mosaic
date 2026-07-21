import hashlib
from pathlib import Path

from mosaic.cache import cache_dir


def test_default_cache_dir(monkeypatch):
    monkeypatch.delenv("MOSAIC_CACHE_DIR", raising=False)

    assert cache_dir() == Path("~/.cache/mosaic").expanduser()


def test_cache_dir_can_be_overridden(monkeypatch, tmp_path):
    root = tmp_path / "models"
    monkeypatch.setenv("MOSAIC_CACHE_DIR", str(root))

    assert cache_dir() == root

def test_opendde_msa_cache_follows_runtime_override(monkeypatch, tmp_path):
    from mosaic.models.opendde import _target_a3m

    sequence = "ACDE"
    digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]

    for name in ("first", "second"):
        root = tmp_path / name
        monkeypatch.setenv("MOSAIC_CACHE_DIR", str(root))
        cached = root / "msa" / f"{digest}.a3m"
        cached.parent.mkdir(parents=True)
        cached.write_text(f">query\n{sequence}\n")

        assert _target_a3m(sequence) == str(cached)


def test_esmfold_msa_cache_follows_runtime_override(monkeypatch, tmp_path):
    from mosaic.models.esmfold2 import _fetch_msa

    sequence = "ACDE"
    digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]

    for name in ("first", "second"):
        root = tmp_path / name
        monkeypatch.setenv("MOSAIC_CACHE_DIR", str(root))
        cached = root / "msa" / f"{digest}.a3m"
        cached.parent.mkdir(parents=True)
        cached.write_text(f">query\n{sequence}\n")

        _fetch_msa(sequence)
        assert cached.exists()
