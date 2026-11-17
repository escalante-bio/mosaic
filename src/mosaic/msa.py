"""Fetch unpaired MSAs from the ColabFold server, caching them on disk."""

import hashlib
import io
import random
import tarfile
import time

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
        a3m.write_text(search(sequence))
    return a3m.read_text()


def search(sequence: str, *, host: str = MSA_SERVER) -> str:
    """Run an unpaired ColabFold search and return the query's a3m text.

    The search covers UniRef and the environmental databases."""
    with httpx.Client(base_url=host, headers={"User-Agent": "mosaic"}, timeout=30.0) as client:
        job = _submit(client, sequence)
        while job["status"] in ("UNKNOWN", "RUNNING", "PENDING"):
            time.sleep(5 + random.randint(0, 5))
            job = _request(client, "GET", f"/ticket/{job['id']}")
        if job["status"] != "COMPLETE":
            raise RuntimeError(f"ColabFold MSA search failed: {job}")
        archive = _request(client, "GET", f"/result/download/{job['id']}", json=False)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        texts = [tar.extractfile(name).read().decode() for name in A3M_FILES]
    return "".join(_query_records(text) for text in texts)


def _submit(client: httpx.Client, sequence: str) -> dict:
    """Submit the search, resubmitting while the server is busy."""
    while True:
        job = _request(
            client, "POST", "/ticket/msa", data={"q": f">{QUERY_ID}\n{sequence}\n", "mode": "env"}
        )
        if job["status"] not in ("UNKNOWN", "RATELIMIT"):
            break
        time.sleep(5 + random.randint(0, 5))
    if job["status"] in ("ERROR", "MAINTENANCE"):
        raise RuntimeError(f"ColabFold MSA server: {job['status']}; check the sequence or retry later")
    return job


def _request(client: httpx.Client, method: str, url: str, *, json: bool = True, **kwargs):
    """Send a request, retrying up to 5 times on network errors.

    Returns the JSON reply, or the raw bytes if `json` is False."""
    for attempt in range(6):
        try:
            response = client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json() if json else response.content
        except httpx.HTTPError:
            if attempt == 5:
                raise
            time.sleep(5)


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
