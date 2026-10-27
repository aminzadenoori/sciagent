"""Download and load link-prediction benchmarks (WN18, WN18RR, FB15k-237)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/villmow/datasets_knowledge_embedding/master"
SOURCES = {
    "WN18": ("WN18/original", ["relation2id.txt"]),
    "WN18RR": ("WN18RR/original", []),
    "FB15k-237": ("FB15k-237", []),
}
SPLITS = ("train", "valid", "test")


def download(name: str, root: Path) -> Path:
    sub, extra = SOURCES[name]
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    for f in [f"{s}.txt" for s in SPLITS] + extra:
        if not (out / f).exists():
            urllib.request.urlretrieve(f"{BASE}/{sub}/{f}", out / f)
    return out


def _read(path: Path, rel_names: dict[str, str] | None):
    lines = path.read_text().splitlines()
    triples = []
    # OpenKE format: first line is a count, then "head tail relation" ids.
    if lines and lines[0].strip().isdigit():
        for ln in lines[1:]:
            h, t, r = ln.split()
            triples.append((h, rel_names.get(r, r) if rel_names else r, t))
    else:
        for ln in lines:
            if ln.strip():
                h, r, t = ln.split("\t")
                triples.append((h, r, t))
    return triples


def load(name: str, root: Path, auto_download: bool = True) -> dict[str, list[tuple[str, str, str]]]:
    d = root / name
    if not (d / "test.txt").exists():
        if not auto_download:
            raise FileNotFoundError(f"{d} missing; run `sciagent kg-download`")
        download(name, root)
    rel_names = None
    if (d / "relation2id.txt").exists():
        rel_names = {}
        for ln in (d / "relation2id.txt").read_text().splitlines()[1:]:
            n, i = ln.split()
            rel_names[i] = n
    return {s: _read(d / f"{s}.txt", rel_names) for s in SPLITS}
