"""Literature-review sub-agent: given a research question, find, screen, download and read papers.

The main ResearchAgent starts it through the `research_topic` tool when a question
is about a topic the corpus does not cover, e.g.

    "What is the highest F1 score for requirements classification with LLMs on the PROMISE dataset?"

Phase 1  search    several queries on arXiv + OpenAlex (synonyms, dataset names)
Phase 2  screen    decide from titles/abstracts which papers report the asked quantity
Phase 3  collect   download the best candidates (capped), ingest text/tables/figures
Phase 4  extract   read results sections/tables and return findings with citations

It runs its own Claude tool loop with a restricted tool set and returns structured
findings (metric, value, dataset, setting, chunk ids) to the main agent, which
compares them and writes the answer. Limits enforced in code: step budget, a
download cap, at least one outside search before submitting, and findings may
only cite chunks the sub-agent actually read (others are dropped and reported).
"""
from __future__ import annotations

import json
import time

from .tools import TOOLS, ToolContext, call_tool

LIT_TOOLS = ["search_external_literature", "fetch_paper", "search_papers", "read_chunks", "get_paper_outline",
             "inspect_figure", "run_python"]

LIT_SYSTEM = """You are a literature-review agent. You receive one research question and must gather evidence from \
papers. Work in phases:

1. Search: call search_external_literature with 2-4 different queries (task name, synonyms, dataset name, method \
family such as "LLM", "BERT", "zero-shot"). Results show title, year, abstract and `fetchable`.
2. Screen: from titles and abstracts, keep only papers likely to report the asked quantity on the asked dataset/task. \
Prefer recent papers and papers whose abstract mentions the dataset and metric.
3. Collect: call fetch_paper on the best candidates (you may download at most {max_papers}). Skip papers already in \
the corpus (already_in_corpus=true): search them directly.
4. Extract: for each paper, use search_papers with paper_ids=[paper_id] (and kinds=["table"] for result tables) to \
find the exact numbers. Read full chunks if truncated.

Then call submit_findings. Rules:
- Every finding must cite the chunk_ids where the number appears. Copy numbers exactly.
- Record the setting that makes numbers comparable or not: dataset version/split (e.g. PROMISE vs PROMISE_exp), task \
variant (binary FR/NFR vs multi-class NFR), metric averaging (macro/weighted), zero-shot/few-shot/fine-tuned, model.
- If you could not download or find the number in a paper, list it under papers_rejected with the reason.
- Do not use knowledge from memory as a finding."""

SUBMIT_FINDINGS = {
    "name": "submit_findings",
    "description": "Return the literature review results. Call exactly once, at the end.",
    "input_schema": {"type": "object", "properties": {
        "summary": {"type": "string", "description": "2-4 sentences: what was searched and what was found."},
        "findings": {"type": "array", "items": {"type": "object", "properties": {
            "paper_id": {"type": "string"}, "finding": {"type": "string"},
            "metric": {"type": "string"}, "value": {"type": "string"}, "dataset": {"type": "string"},
            "setting": {"type": "string", "description": "task variant, split, model, zero/few-shot or fine-tuned"},
            "chunk_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["paper_id", "finding", "chunk_ids"]}},
        "papers_rejected": {"type": "array", "items": {"type": "object", "properties": {
            "title": {"type": "string"}, "reason": {"type": "string"}}}},
        "gaps": {"type": "string", "description": "What could not be established and why."}},
        "required": ["summary", "findings"]}}


def _blocks(resp):
    from .agent import _block_to_dict
    return [_block_to_dict(b) for b in resp.content]


class LiteratureAgent:
    def __init__(self, ctx: ToolContext, max_papers: int = 5, max_steps: int = 16, emit=None):
        self.ctx, self.max_papers, self.max_steps = ctx, max(1, min(int(max_papers), 10)), max_steps
        self.emit = emit or (lambda e: None)

    def _specs(self):
        specs = [{"name": n, **TOOLS[n][1]} for n in LIT_TOOLS] + [SUBMIT_FINDINGS]
        specs[-1] = {**specs[-1], "cache_control": {"type": "ephemeral"}}
        return specs

    def run(self, question: str) -> dict:
        ctx, t0 = self.ctx, time.time()
        system = LIT_SYSTEM.replace("{max_papers}", str(self.max_papers))
        messages = [{"role": "user", "content": f"Research question: {question}"}]
        fetched, steps, events, rejected_once = [], [], [], set()
        usage = {"input_tokens": 0, "output_tokens": 0}
        out = None
        for step in range(self.max_steps + 1):
            force = step == self.max_steps
            if force:
                events.append("step_budget_exhausted")
                messages.append({"role": "user", "content": "Step budget used up. Call submit_findings now."})
            kw = dict(model=ctx.model, max_tokens=2048, tools=self._specs(), messages=messages,
                      system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}])
            if force:
                kw["tool_choice"] = {"type": "tool", "name": "submit_findings"}
            resp = ctx.client.messages.create(**kw)
            u = getattr(resp, "usage", None)
            if u:
                usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
                usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
            blocks = _blocks(resp)
            messages.append({"role": "assistant", "content": blocks})
            uses = [b for b in blocks if b.get("type") == "tool_use"]
            if not uses:
                messages.append({"role": "user", "content": "Use the tools, and finish with submit_findings."})
                continue
            results, final = [], None
            for tu in uses:
                name, args = tu["name"], tu.get("input", {})
                if name == "submit_findings":
                    final = tu
                    continue
                if name not in LIT_TOOLS:
                    res = json.dumps({"error": f"{name} is not available to the literature agent"})
                elif name == "fetch_paper" and len(fetched) >= self.max_papers:
                    res = json.dumps({"error": f"download budget of {self.max_papers} papers used; extract from "
                                               f"the papers you have"})
                else:
                    self.emit({"type": "tool_call", "tool": name, "args": args, "depth": 1})
                    ts = time.time()
                    res = call_tool(ctx, name, args)
                    self.emit({"type": "tool_result", "tool": name, "output": res[:4000], "depth": 1,
                               "seconds": round(time.time() - ts, 2)})
                    if name == "fetch_paper" and '"paper_id"' in res:
                        fetched.append(json.loads(res)["paper_id"])
                steps.append({"tool": name, "args": args, "output_preview": res[:300]})
                results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": res})
            if final is not None:
                problem = None
                if (ctx.allow_network and "outside" not in rejected_once
                        and not any(s["tool"] == "search_external_literature" for s in steps)):
                    problem = "outside"
                    msg = "Search outside the corpus with search_external_literature before submitting findings."
                if problem:
                    rejected_once.add(problem)
                    events.append("rejected: no outside search")
                    results.append({"type": "tool_result", "tool_use_id": final["id"], "is_error": True, "content": msg})
                    messages.append({"role": "user", "content": results})
                    continue
                out = final.get("input", {})
                break
            messages.append({"role": "user", "content": results})
        out = out or {"summary": "The literature agent stopped without findings.", "findings": []}
        valid, dropped = [], []
        for f in out.get("findings", []) or []:
            ok = [c for c in f.get("chunk_ids", []) or [] if c in ctx.seen_chunks]
            bad = [c for c in f.get("chunk_ids", []) or [] if c not in ctx.seen_chunks]
            if bad:
                dropped += bad
            if ok:
                valid.append({**f, "chunk_ids": ok})
        if dropped:
            events.append("unverified_finding_citations_dropped")
        return {"summary": out.get("summary", ""), "findings": valid,
                "papers_fetched": fetched, "papers_rejected": out.get("papers_rejected", []) or [],
                "gaps": out.get("gaps", ""), "dropped_citations": dropped, "events": events,
                "tool_calls": [s["tool"] for s in steps], "usage": usage, "seconds": round(time.time() - t0, 2),
                "note": "Findings are verified to cite chunks the literature agent read. Compare values only when "
                        "dataset, task variant and metric match; cite the chunk_ids."}
