"""Fetch unpaired MSAs from the ColabFold server, caching them on disk."""

import hashlib
import io
import math
import random
import tarfile
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx

from mosaic.cache import cache_dir

MSA_SERVER = "https://api.colabfold.com"
# The result files to read, in the order they are concatenated.
A3M_FILES = ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m")
QUERY_ID = 101  # the id ColabFold's client gives the first query


def fetch_a3m(sequence: str) -> str:
    """Return the unpaired ColabFold MSA for `sequence` as a3m text.

    Results are cached by sequence hash in the `msa` cache directory."""
    msa_cache = cache_dir("msa")
    msa_cache.mkdir(parents=True, exist_ok=True)
    a3m = msa_cache / f"{hashlib.sha256(sequence.encode()).hexdigest()[:16]}.a3m"
    if not a3m.exists():
        result = search(sequence)
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", dir=msa_cache, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(result)
            temporary.replace(a3m)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return a3m.read_text()


def search(sequence: str, *, host: str = MSA_SERVER, timeout: float = 1800) -> str:
    """Run an unpaired ColabFold search and return the query's a3m text.

    The monotonic timeout (seconds) covers submission, retries, polling and download.
    HTTPX timeouts bound individual network operations. A continuously streaming
    response can exceed the deadline before control returns to this client."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    deadline = time.monotonic() + timeout
    with httpx.Client(
        base_url=host, headers={"User-Agent": "mosaic"}, timeout=30.0
    ) as client:
        job = _submit(client, sequence, deadline)
        while job["status"] in ("UNKNOWN", "RUNNING", "PENDING"):
            _sleep(5 + random.randint(0, 5), deadline)
            job = _request(client, deadline, "GET", f"/ticket/{job['id']}")
        if job["status"] != "COMPLETE":
            raise RuntimeError(f"ColabFold MSA search failed: {job}")
        archive = _request(
            client, deadline, "GET", f"/result/download/{job['id']}", json=False
        )
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        texts = [tar.extractfile(name).read().decode() for name in A3M_FILES]
    result = "".join(_query_records(text) for text in texts)
    _remaining(deadline)
    return result


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("ColabFold MSA search deadline exceeded")
    return remaining


def _sleep(seconds: float, deadline: float) -> None:
    time.sleep(min(seconds, _remaining(deadline)))


def _submit(client: httpx.Client, sequence: str, deadline: float) -> dict:
    """Submit the search, resubmitting while the server is busy."""
    while True:
        job = _request(
            client,
            deadline,
            "POST",
            "/ticket/msa",
            data={"q": f">{QUERY_ID}\n{sequence}\n", "mode": "env"},
        )
        if job["status"] not in ("UNKNOWN", "RATELIMIT"):
            break
        _sleep(5 + random.randint(0, 5), deadline)
    if job["status"] in ("ERROR", "MAINTENANCE"):
        raise RuntimeError(
            f"ColabFold MSA server: {job['status']}; check the sequence or retry later"
        )
    return job


def _request(
    client: httpx.Client,
    deadline: float,
    method: str,
    url: str,
    *,
    json: bool = True,
    **kwargs,
):
    """Send a request, retrying up to 5 times on network errors.

    Returns the JSON reply, or the raw bytes if `json` is False."""
    for attempt in range(6):
        try:
            response = client.request(
                method, url, timeout=min(30, _remaining(deadline)), **kwargs
            )
            _remaining(deadline)
            response.raise_for_status()
            return response.json() if json else response.content
        except httpx.HTTPError:
            _remaining(deadline)
            if attempt == 5:
                raise
            _sleep(5, deadline)


def _query_records(text: str) -> str:
    """Return the query's records from one a3m result file.

    In these files, each query's alignment starts after a NUL byte and a `>id` line."""
    records, query = [], None
    new_query = True
    for line in text.splitlines(keepends=True):
        if "\x00" in line:
            line = line.replace("\x00", "")
            new_query = True
        if line.startswith(">") and new_query:
            query = int(line[1:].rstrip())
            new_query = False
        if query == QUERY_ID:
            records.append(line)
    return "".join(records)
