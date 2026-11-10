"""MSAs from the ColabFold server, shared by every backend: unpaired a3m text per sequence,
cached on disk so each sequence is searched once."""

import hashlib
import io
import random
import tarfile
import time

import httpx

from mosaic.cache import cache_dir

MSA_SERVER = "https://api.colabfold.com"
# The search's a3m files, joined in this order: UniRef, then the environmental databases.
A3M_FILES = ("uniref.a3m", "bfd.mgnify30.metaeuk30.smag30.a3m")
QUERY_ID = 101  # the name ColabFold's client gives the (first) query


def fetch_a3m(sequence: str) -> str:
    """The unpaired ColabFold MSA (a3m text) for `sequence`, as ColabFold's client (and
    boltz's) fetches it, cached by sequence hash under `MOSAIC_CACHE_DIR/msa`."""
    msa_cache = cache_dir("msa")
    msa_cache.mkdir(parents=True, exist_ok=True)
    a3m = msa_cache / f"{hashlib.sha256(sequence.encode()).hexdigest()[:16]}.a3m"
    if not a3m.exists():
        a3m.write_text(search(sequence))
    return a3m.read_text()


def search(sequence: str, *, host: str = MSA_SERVER) -> str:
    """One unpaired search (UniRef and environmental databases, filtered) on the ColabFold
    server: submit, wait for the job, download the results; the query's a3m text."""
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
    """`method url`, retried up to 5 times on network errors; the JSON reply (or bytes)."""
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
    """The query's records in one of the search's a3m files, which hold each query's
    alignment after a NUL byte and a `>id` line."""
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
