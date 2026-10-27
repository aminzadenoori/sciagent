"""Tools the agent can call. Each tool = JSON schema (sent to Claude) + a Python function.

Tool outputs are plain JSON strings. Every evidence-bearing output carries
`chunk_id`s; the agent may only cite chunk_ids that some tool actually returned
in the current session (enforced in agent.py), which makes fabricated
citations detectable.
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

from .index import HybridIndex

MAX_TEXT = 900


class ToolContext:
    def __init__(self, index: HybridIndex, data_dir: Path, client=None, model: str | None = None, allow_network: bool = True):
        self.index = index
        self.data_dir = data_dir
        self.client = client
        self.model = model
        self.allow_network = allow_network
        self.seen_chunks: set[str] = set()  # chunk ids returned to the model this session
        self.focus_paper: str | None = None  # when set, paper searches are restricted to this paper

    def _emit(self, hits) -> list[dict]:
        out = []
        for h in hits:
            self.seen_chunks.add(h.chunk_id)
            p = self.index.papers.get(h.paper_id, {})
            out.append({"chunk_id": h.chunk_id, "paper_id": h.paper_id, "paper_title": p.get("title"),
                        "year": p.get("year"), "section": h.section, "kind": h.kind,
                        "text": h.text if len(h.text) <= MAX_TEXT else h.text[:MAX_TEXT] + " ...[truncated; use read_chunks]"})
        return out


# ------------------------------------------------------------------ implementations
def search_papers(ctx: ToolContext, query: str, k: int = 6, kinds: list[str] | None = None,
                  paper_ids: list[str] | None = None) -> dict:
    k = max(1, min(int(k), 15))
    if ctx.focus_paper and not paper_ids:
        paper_ids = [ctx.focus_paper]
    hits = ctx.index.search(query, k=k, kinds=kinds, paper_ids=paper_ids, per_paper=k if paper_ids and len(paper_ids) == 1 else 3)
    return {"query": query, "n": len(hits), "results": ctx._emit(hits),
            "note": None if hits else "No matching evidence in the indexed corpus."}


def read_chunks(ctx: ToolContext, chunk_ids: list[str]) -> dict:
    out, missing = [], []
    for cid in chunk_ids[:10]:
        c = ctx.index.by_id.get(cid)
        if not c:
            missing.append(cid)
            continue
        ctx.seen_chunks.add(cid)
        out.append({"chunk_id": cid, "paper_id": c["paper_id"], "section": c["section"], "kind": c["kind"], "text": c["text"]})
    return {"chunks": out, "missing": missing}


def list_papers(ctx: ToolContext, author: str | None = None, year_from: int | None = None, query: str | None = None) -> dict:
    rows = []
    for p in ctx.index.papers.values():
        if author and author.lower() not in " ".join(p.get("authors", [])).lower():
            continue
        if year_from and (p.get("year") or 0) < year_from:
            continue
        if query and query.lower() not in p.get("title", "").lower():
            continue
        ctx.seen_chunks.add(f"{p['id']}#meta")
        rows.append({"cite_as": f"{p['id']}#meta", "paper_id": p["id"], "title": p["title"], "year": p.get("year"), "authors": p.get("authors", [])[:8],
                     "venue": p.get("venue"), "source": p.get("source")})
    rows.sort(key=lambda r: -(r["year"] or 0))
    return {"n": len(rows), "papers": rows}


def get_paper_outline(ctx: ToolContext, paper_id: str) -> dict:
    p = ctx.index.papers.get(paper_id)
    if not p:
        return {"error": f"unknown paper_id {paper_id!r}; call list_papers to see valid ids"}
    chunks = [c for c in ctx.index.chunks if c["paper_id"] == paper_id]
    ctx.seen_chunks.add(f"{paper_id}#meta")
    return {"cite_as": f"{paper_id}#meta", "paper_id": paper_id, "title": p["title"], "authors": p.get("authors"), "year": p.get("year"),
            "venue": p.get("venue"), "url": p.get("url"),
            "parts": [{"chunk_id": c["chunk_id"], "kind": c["kind"], "section": c["section"],
                       "caption": c.get("meta", {}).get("caption")} for c in chunks],
            "note": "card = structured summary, not full text" if p.get("sections") else None}


def inspect_figure(ctx: ToolContext, chunk_id: str, question: str) -> dict:
    c = ctx.index.by_id.get(chunk_id)
    if not c or c["kind"] != "figure":
        return {"error": f"{chunk_id!r} is not a figure chunk; search with kinds=['figure'] first"}
    ctx.seen_chunks.add(chunk_id)
    img = Path(c["meta"].get("image_path", ""))
    caption = c["meta"].get("caption")
    if not img.exists():
        return {"chunk_id": chunk_id, "caption": caption, "error": "image file missing; only caption available"}
    if ctx.client is None:
        return {"chunk_id": chunk_id, "caption": caption, "note": "vision model unavailable; caption only"}
    b64 = base64.standard_b64encode(img.read_bytes()).decode()
    resp = ctx.client.messages.create(
        model=ctx.model, max_tokens=700,
        system="You read scientific figures. Report only what is visible: axes, units, series, values you can read "
               "(with approximate precision), and trends. Say explicitly when something cannot be read from the image.",
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": f"Caption: {caption}\n\nQuestion: {question}"}]}])
    text = "".join(getattr(b, "text", "") for b in resp.content)
    return {"chunk_id": chunk_id, "caption": caption, "vision_reading": text}


def run_python(ctx: ToolContext, code: str, timeout: int = 60) -> dict:
    """Run code in a separate process (cwd = data dir). Not a security sandbox: local prototype only."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run([sys.executable, "-I", path], cwd=ctx.data_dir, capture_output=True, text=True,
                           timeout=min(int(timeout), 120), env={**os.environ, "MPLBACKEND": "Agg"})
        return {"returncode": r.returncode, "stdout": r.stdout[-4000:], "stderr": r.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout}s"}
    finally:
        os.unlink(path)


def kg_leakage_analysis(ctx: ToolContext, dataset: str, min_conf: float = 0.9) -> dict:
    from .kg.analysis import inverse_rule_eval, leakage_stats
    from .kg.data import SOURCES, load
    if dataset not in SOURCES:
        return {"error": f"unknown dataset; choose one of {list(SOURCES)}"}
    try:
        split = load(dataset, ctx.data_dir / "kg", auto_download=ctx.allow_network)
    except Exception as e:
        return {"error": f"could not load {dataset}: {e}"}
    return {"dataset": dataset, "leakage": leakage_stats(split, min_conf),
            "rule_model": inverse_rule_eval(split, min_conf),
            "note": "computed now from the benchmark files; cite as tool output, not as a paper"}


def search_external_literature(ctx: ToolContext, query: str, n: int = 8) -> dict:
    if not ctx.allow_network:
        return {"error": "network disabled"}
    from .collect import openalex_search
    try:
        recs = openalex_search(query, min(int(n), 15))
    except Exception as e:
        return {"error": f"external search failed: {e}"}
    return {"note": "NOT in the indexed corpus: abstract-level metadata only; use for discovery, not as primary evidence",
            "results": [{"title": r["title"], "year": r["year"], "authors": r["authors"][:5], "venue": r.get("venue"),
                         "cited_by": r.get("cited_by"), "id": r["id"], "abstract": (r.get("abstract") or "")[:600]}
                        for r in recs]}


_SCIQA = {}


def query_scholarly_kg(ctx: ToolContext, question: str, shots: int = 3) -> dict:
    """NL -> SPARQL over ORKG with retrieved SciQA train examples (no fine-tuning), then execute."""
    import hashlib
    from .sciqa import ClaudeSparql, ExampleRetriever, download, execute, load_split
    if ctx.client is None:
        return {"error": "LLM client unavailable"}
    if "retriever" not in _SCIQA:
        root = ctx.data_dir / "sciqa"
        if not (root / "train" / "questions.json").exists():
            if not ctx.allow_network:
                return {"error": "SciQA example pool missing (run evals/run_sciqa.py once to download it)"}
            download(root)
        _SCIQA["retriever"] = ExampleRetriever(load_split(root, "train"))
    gen = ClaudeSparql(_SCIQA["retriever"], client=ctx.client, model=ctx.model, k=max(0, min(int(shots), 5)))
    sparql = gen(question)
    answers = execute(sparql) if ctx.allow_network else None
    cid = "orkg:" + hashlib.sha1(sparql.encode()).hexdigest()[:10] + "#sparql"
    ctx.seen_chunks.add(cid)
    return {"cite_as": cid, "sparql": sparql,
            "examples_used": [ex.question for ex, _ in _SCIQA["retriever"].top_k(question, gen.k)],
            "answers": answers[:50] if answers else answers, "n_answers": None if answers is None else len(answers),
            "status": "executed" if answers is not None else "execution failed or endpoint unreachable",
            "note": "ORKG content changes over time; an empty result can mean a wrong query OR missing data"}


# ------------------------------------------------------------------ schemas
TOOLS: dict[str, tuple[Callable, dict]] = {
    "search_papers": (search_papers, {
        "description": "Hybrid (BM25 + dense) search over the indexed corpus of papers. Returns evidence chunks with "
                       "chunk_ids to cite. Use several focused queries for multi-part questions. kinds filters to "
                       "'text', 'table' or 'figure' (figure chunks contain captions).",
        "input_schema": {"type": "object", "properties": {
            "query": {"type": "string"}, "k": {"type": "integer", "default": 6},
            "kinds": {"type": "array", "items": {"type": "string", "enum": ["text", "table", "figure"]}},
            "paper_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["query"]}}),
    "read_chunks": (read_chunks, {
        "description": "Read the full text of chunks by chunk_id (search results may be truncated).",
        "input_schema": {"type": "object", "properties": {"chunk_ids": {"type": "array", "items": {"type": "string"}}},
                         "required": ["chunk_ids"]}}),
    "list_papers": (list_papers, {
        "description": "List papers in the corpus (id, title, year, authors), optionally filtered by author name, "
                       "minimum year or title substring. Use for questions about what the corpus or an author covers. "
                       "Cite bibliographic facts as [<paper_id>#meta].",
        "input_schema": {"type": "object", "properties": {"author": {"type": "string"}, "year_from": {"type": "integer"},
                                                          "query": {"type": "string"}}}}),
    "get_paper_outline": (get_paper_outline, {
        "description": "Metadata and the list of sections, tables and figures (with chunk_ids and captions) of one paper.",
        "input_schema": {"type": "object", "properties": {"paper_id": {"type": "string"}}, "required": ["paper_id"]}}),
    "inspect_figure": (inspect_figure, {
        "description": "Look at a figure image with a vision model and answer a question about it. Use when the "
                       "answer depends on values or trends in a plot that the caption does not state.",
        "input_schema": {"type": "object", "properties": {"chunk_id": {"type": "string"}, "question": {"type": "string"}},
                         "required": ["chunk_id", "question"]}}),
    "run_python": (run_python, {
        "description": "Run Python 3 (numpy, pandas, scipy, sklearn available) for calculations or data analysis. "
                       "Working directory is the data folder (kg/ holds benchmark triples, corpus/kg_leakage_results.json "
                       "holds computed results). Print what you need.",
        "input_schema": {"type": "object", "properties": {"code": {"type": "string"}, "timeout": {"type": "integer"}},
                         "required": ["code"]}}),
    "kg_leakage_analysis": (kg_leakage_analysis, {
        "description": "Compute inverse-relation test leakage and a rule-only baseline (filtered MRR, Hits@10, random "
                       "tie-breaking) on a link-prediction benchmark: WN18, WN18RR or FB15k-237.",
        "input_schema": {"type": "object", "properties": {"dataset": {"type": "string", "enum": ["WN18", "WN18RR", "FB15k-237"]},
                                                          "min_conf": {"type": "number"}}, "required": ["dataset"]}}),
    "query_scholarly_kg": (query_scholarly_kg, {
        "description": "Answer structured questions about research results recorded in the Open Research Knowledge "
                       "Graph (ORKG), e.g. which models were evaluated on a dataset, best score on a benchmark, papers "
                       "with a benchmark. Generates SPARQL with retrieved SciQA examples and executes it. Cite as cite_as.",
        "input_schema": {"type": "object", "properties": {"question": {"type": "string"}, "shots": {"type": "integer"}},
                         "required": ["question"]}}),
    "search_external_literature": (search_external_literature, {
        "description": "Search OpenAlex for papers NOT in the corpus (titles + abstracts). Use when the corpus lacks "
                       "evidence, to tell the user what exists. Results cannot be cited as chunk evidence.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "n": {"type": "integer"}},
                         "required": ["query"]}}),
}

SUBMIT_ANSWER = {
    "name": "submit_answer",
    "description": "Submit the final answer. Call exactly once, at the end.",
    "input_schema": {"type": "object", "properties": {
        "answer": {"type": "string", "description": "Answer in prose. Put [chunk_id] after each claim it supports."},
        "citations": {"type": "array", "items": {"type": "string"}, "description": "All chunk_ids cited in the answer."},
        "evidence_status": {"type": "string", "enum": ["supported", "partially_supported", "conflicting", "insufficient"],
                            "description": "conflicting = papers in the corpus disagree; insufficient = corpus does not answer it."},
        "confidence": {"type": "number", "description": "0-1"},
        "conflicts": {"type": "array", "items": {"type": "string"},
                      "description": "One line per disagreement: who claims what vs who claims what."}},
        "required": ["answer", "citations", "evidence_status", "confidence"]}}


def tool_specs() -> list[dict]:
    return [{"name": n, **spec} for n, (_, spec) in TOOLS.items()] + [SUBMIT_ANSWER]


def call_tool(ctx: ToolContext, name: str, args: dict[str, Any]) -> str:
    if name not in TOOLS:
        return json.dumps({"error": f"unknown tool {name}"})
    fn = TOOLS[name][0]
    try:
        return json.dumps(fn(ctx, **args), default=str)
    except TypeError as e:
        return json.dumps({"error": f"bad arguments for {name}: {e}"})
    except Exception as e:  # a failing tool must not kill the agent loop
        return json.dumps({"error": f"{name} failed: {type(e).__name__}: {e}"})
