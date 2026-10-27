"""Command line: python -m sciagent <command>.

  collect   find + download papers (arXiv/OpenAlex), seeded by author and topics
  analyze   download KG benchmarks, run leakage analysis, write report PDF into the corpus
  build     ingest corpus (PDF text/tables/figures, cards) and build the hybrid index
  search    retrieval only (no LLM) - handy for debugging
  ask       run the Claude agent on a question
  add       add one paper (PDF file, arXiv id or PDF link) and re-index
  serve     local web interface on http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

DATA = Path(os.environ.get("SCIAGENT_DATA", "data"))
CORPUS, INDEX, KG = DATA / "corpus", DATA / "index", DATA / "kg"
SEED = Path(__file__).resolve().parent.parent / "data" / "seed_corpus" / "cards.jsonl"


def cmd_collect(a):
    from .collect import CollectConfig, collect
    cfg = CollectConfig(author=a.author, download_pdfs=not a.no_pdfs, max_per_query=a.max)
    if a.topic:
        cfg.topics = a.topic
    recs = collect(cfg, CORPUS)
    print(f"collected {len(recs)} papers -> {CORPUS / 'collected.jsonl'}")


def cmd_analyze(a):
    from .kg.report import run, write_report
    res = run(KG)
    path = write_report(res, CORPUS)
    print(json.dumps({d: {"recoverable_pct": r["leakage"]["test_recoverable_by_rule_pct"], "rule_MRR": r["rule_model"]["MRR"]}
                      for d, r in res.items()}, indent=1))
    print(f"report -> {path}")


def cmd_build(a):
    from .index import HybridIndex
    from .ingest import ingest_corpus
    CORPUS.mkdir(parents=True, exist_ok=True)
    if not a.no_seed and SEED.exists():
        shutil.copy(SEED, CORPUS / "cards.jsonl")
    rep = ingest_corpus(CORPUS, INDEX)
    idx = HybridIndex.build(INDEX)
    kinds = {}
    for c in idx.chunks:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    print(f"papers: {len(idx.papers)}  chunks: {kinds}")
    bad = {k: v for k, v in rep.items() if v.get("needs_ocr") or v.get("error") or v.get("abstract_only")}
    if bad:
        print("degraded records:", json.dumps(bad, indent=1))


def cmd_search(a):
    from .index import HybridIndex
    idx = HybridIndex.load(INDEX)
    for h in idx.search(a.query, k=a.k, mode=a.mode, kinds=a.kinds):
        print(f"{h.score:.4f}  {h.chunk_id:45s} {h.text[:110]!r}")


def cmd_ask(a):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Export it, or use `python -m sciagent search` for retrieval only.")
    from .agent import ResearchAgent
    from .index import HybridIndex
    idx = HybridIndex.load(INDEX)
    agent = ResearchAgent(idx, DATA, model=a.model, max_steps=a.max_steps, log=print if a.verbose else None)
    r = agent.run(a.question, focus_paper=a.paper)
    if a.json:
        print(json.dumps(r.to_dict(), indent=1))
        return
    print(f"\n[{r.evidence_status} | confidence {r.confidence:.2f} | {len(r.steps)} tool calls | {r.seconds}s]\n")
    print(r.answer)
    for c in r.conflicts:
        print(" - conflict:", c)
    print("\nCited:", ", ".join(r.citations) or "-")
    if r.guardrail_events or r.error:
        print("Guardrails:", r.guardrail_events, r.error or "")


def cmd_add(a):
    from .library import add_paper
    out = add_paper(a.source, CORPUS, INDEX, title=a.title)
    print(f"added {out['paper_id']}: {out['title']}")
    print(f"indexed parts: {out['chunks']}  {out['report']}")
    print(f'ask about it:  python -m sciagent ask "..." --paper {out["paper_id"]}')


def cmd_serve(a):
    from .web.server import serve
    serve(DATA, port=a.port, model=a.model)


def main(argv=None):
    p = argparse.ArgumentParser(prog="sciagent")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("collect"); s.add_argument("--author", default="Sahar Vahdati")
    s.add_argument("--topic", action="append"); s.add_argument("--max", type=int, default=25)
    s.add_argument("--no-pdfs", action="store_true"); s.set_defaults(f=cmd_collect)
    s = sub.add_parser("analyze"); s.set_defaults(f=cmd_analyze)
    s = sub.add_parser("build"); s.add_argument("--no-seed", action="store_true"); s.set_defaults(f=cmd_build)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=8)
    s.add_argument("--mode", default="hybrid", choices=["hybrid", "bm25", "dense"])
    s.add_argument("--kinds", nargs="*"); s.set_defaults(f=cmd_search)
    s = sub.add_parser("ask"); s.add_argument("question")
    s.add_argument("--model", default=os.environ.get("SCIAGENT_MODEL", "claude-sonnet-5-5"))
    s.add_argument("--max-steps", type=int, default=12); s.add_argument("--json", action="store_true")
    s.add_argument("-v", "--verbose", action="store_true")
    s.add_argument("--paper", help="answer only from this paper id (see the web page or list in data/index/papers.jsonl)")
    s.set_defaults(f=cmd_ask)
    s = sub.add_parser("add"); s.add_argument("source", help="PDF path, arXiv id/URL, or direct PDF URL")
    s.add_argument("--title"); s.set_defaults(f=cmd_add)
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8000)
    s.add_argument("--model", default=os.environ.get("SCIAGENT_MODEL", "claude-sonnet-5-5")); s.set_defaults(f=cmd_serve)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main(sys.argv[1:])
