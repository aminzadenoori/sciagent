"""The research agent: a Claude tool-use loop with evidence guardrails.

Claude decides which tools to call and in what order (no fixed pipeline). The
loop only enforces things a model should not be trusted to self-police:

* step budget (max_steps) and a forced final answer when it runs out;
* the answer must come through `submit_answer` (structured output);
* citation validation: every cited chunk_id must exist AND must have been
  returned by a tool in this session. Invalid citations are sent back once for
  self-correction; if they persist they are stripped and the answer is
  downgraded, never silently accepted;
* "insufficient" answers with zero retrieval are rejected once (the model must
  look before it abstains), and, with network access, so are ones that never
  searched outside the corpus (search_external_literature or research_topic).

Sub-agent: `research_topic` starts the LiteratureAgent (subagents.py), which runs
its own tool loop to find, screen, download and read papers; the chunks it read
are shared with this agent so its findings can be cited here.

Everything is recorded in a trace for evaluation and debugging.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .index import HybridIndex
from .tools import ToolContext, call_tool, tool_specs

DEFAULT_MODEL = os.environ.get("SCIAGENT_MODEL", "claude-sonnet-5-5")

SYSTEM = """You are a careful research assistant answering questions about a corpus of scientific papers \
(knowledge-graph embeddings, link prediction, knowledge graphs + LLMs, and related work), a scholarly knowledge graph (ORKG, via query_scholarly_kg) and data-analysis tools.

How to work:
- Decide which tools you need. Search with several focused queries for multi-part questions; read full chunks when \
results are truncated; inspect figures when values must be read from a plot; use run_python or kg_leakage_analysis \
for computations instead of doing arithmetic in your head.
- Ground every factual claim in evidence returned by tools and cite it as [chunk_id]. Never cite a chunk you have \
not seen. Numbers must match the source exactly; say so when you round.
- Robustness rules:
  * If papers disagree, do not pick a winner silently: set evidence_status="conflicting", list each side with \
citations in `conflicts`, and explain what drives the disagreement (e.g. different assumptions, protocols, datasets).
  * If the question is about a specific paper that is not in the corpus (check with list_papers or search_papers), \
or the corpus does not contain the answer: call search_external_literature, pick the result whose title matches \
(title_match close to 1), add it with fetch_paper, then search_papers with paper_ids=[its paper_id] and answer from \
it with citations. Say in the answer that the paper was added from outside the corpus. Only if no matching, \
downloadable paper exists set evidence_status="insufficient", list the closest candidates you found, and do not fill \
the gap from memory.
  * If the question is about a research topic or the state of the art (e.g. "what is the highest F1 for task X \
on dataset Y"), first check the corpus with search_papers. If it lacks papers that report this, call research_topic: \
a literature-review agent that searches, downloads and reads papers and returns findings with chunk_ids. Then \
compare the findings yourself: name the best reported value with its paper, model and setting, and say which \
numbers are NOT comparable (different dataset version, task variant, metric averaging, zero-shot vs fine-tuned). \
If the top results come from incomparable settings, use evidence_status="conflicting" or "partially_supported". \
Make clear that "highest" means highest among the papers found, not necessarily in all literature.
  * If the question contains a false premise, correct it with evidence.
  * Some records are structured summaries ("cards") rather than full text; mention this if precision matters.
- Finish by calling submit_answer exactly once. Be concise."""


@dataclass
class Step:
    tool: str
    args: dict
    output_preview: str
    seconds: float


@dataclass
class AgentResult:
    question: str
    answer: str = ""
    citations: list = field(default_factory=list)
    evidence_status: str = "insufficient"
    confidence: float = 0.0
    conflicts: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    invalid_citations: list = field(default_factory=list)
    guardrail_events: list = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    seconds: float = 0.0
    error: str | None = None
    added_papers: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _block_to_dict(b) -> dict:
    if hasattr(b, "model_dump"):
        d = b.model_dump(exclude_none=True)
    elif isinstance(b, dict):
        d = dict(b)
    else:
        d = dict(b.__dict__)
    keep = {"text": ("type", "text"), "tool_use": ("type", "id", "name", "input")}
    if d.get("type") in keep:
        return {k: d[k] for k in keep[d["type"]] if k in d}
    return d


class ResearchAgent:
    def __init__(self, index: HybridIndex, data_dir: Path, client=None, model: str = DEFAULT_MODEL,
                 max_steps: int = 12, allow_network: bool = True, log=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.client, self.model, self.max_steps = client, model, max_steps
        self.index, self.data_dir, self.allow_network = index, data_dir, allow_network
        self.log = log or (lambda *a: None)

    def _create(self, messages, tool_choice=None):
        tools = tool_specs()
        tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}  # cache system + tools across steps
        kw = dict(model=self.model, max_tokens=2048, tools=tools, messages=messages,
                  system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}])
        if tool_choice:
            kw["tool_choice"] = tool_choice
        for attempt in range(4):
            try:
                return self.client.messages.create(**kw)
            except Exception as e:  # rate limits / overload: back off and retry
                if attempt == 3 or "rate" not in str(e).lower() and "overload" not in str(e).lower():
                    raise
                time.sleep(2 ** attempt * 2)

    def run(self, question: str, on_event=None, focus_paper: str | None = None) -> AgentResult:
        """on_event(dict) is called live with tool_call / tool_result / guardrail events (used by the web UI)."""
        emit = on_event or (lambda e: None)
        t0 = time.time()
        ctx = ToolContext(self.index, self.data_dir, client=self.client, model=self.model, allow_network=self.allow_network)
        ctx.emit = emit
        res = AgentResult(question=question)
        content = question
        if focus_paper:
            p = self.index.papers.get(focus_paper)
            if p is None:
                res.error = f"unknown paper {focus_paper!r}"
                res.answer = "That paper is not in the corpus."
                return res
            ctx.focus_paper = focus_paper
            content = (f'This question is about the paper "{p["title"]}" (paper_id {focus_paper}). search_papers is '
                       f"restricted to that paper. Answer from it; if it does not contain the answer, say so "
                       f"(evidence_status=insufficient) rather than using other knowledge.\n\nQuestion: {question}")
        messages = [{"role": "user", "content": content}]
        rejected: set[str] = set()  # each kind of guardrail rejection is sent back to the model at most once
        self._focus = focus_paper
        try:
            for step in range(self.max_steps + 1):
                force = step == self.max_steps
                if force:
                    res.guardrail_events.append("step_budget_exhausted")
                    messages.append({"role": "user", "content": "Step budget used up. Call submit_answer now with what you have."})
                resp = self._create(messages, tool_choice={"type": "tool", "name": "submit_answer"} if force else None)
                u = getattr(resp, "usage", None)
                if u:
                    res.usage["input_tokens"] += getattr(u, "input_tokens", 0) or 0
                    res.usage["output_tokens"] += getattr(u, "output_tokens", 0) or 0
                blocks = [_block_to_dict(b) for b in resp.content]
                messages.append({"role": "assistant", "content": blocks})
                uses = [b for b in blocks if b.get("type") == "tool_use"]
                if not uses:  # model answered in plain text: ask for the structured submission
                    res.guardrail_events.append("plain_text_answer")
                    messages.append({"role": "user", "content": "Please submit your answer with the submit_answer tool."})
                    continue
                results, final = [], None
                for u_ in uses:
                    if u_["name"] == "submit_answer":
                        final = u_
                        continue
                    ts = time.time()
                    emit({"type": "tool_call", "tool": u_["name"], "args": u_.get("input", {})})
                    out = call_tool(ctx, u_["name"], u_.get("input", {}))
                    emit({"type": "tool_result", "tool": u_["name"], "output": out[:4000],
                          "seconds": round(time.time() - ts, 2)})
                    res.steps.append(Step(u_["name"], u_.get("input", {}), out[:300], round(time.time() - ts, 2)))
                    self.log(f"  -> {u_['name']}({json.dumps(u_.get('input', {}))[:120]})")
                    results.append({"type": "tool_result", "tool_use_id": u_["id"], "content": out})
                if final is not None:
                    ans = final.get("input", {})
                    problem = self._check(ans, ctx, res)
                    kind = problem.split(" check")[0] if problem else None
                    if problem and kind not in rejected:
                        rejected.add(kind)
                        res.guardrail_events.append(problem)
                        emit({"type": "guardrail", "message": problem})
                        results.append({"type": "tool_result", "tool_use_id": final["id"], "is_error": True,
                                        "content": f"Answer rejected: {problem}. Fix it and call submit_answer again."})
                        messages.append({"role": "user", "content": results})
                        continue
                    self._finalise(ans, ctx, res)
                    break
                messages.append({"role": "user", "content": results})
        except Exception as e:
            res.error = f"{type(e).__name__}: {e}"
            res.answer = res.answer or "The agent failed before producing an answer."
        res.seconds = round(time.time() - t0, 2)
        res.added_papers = list(ctx.added_papers)
        self.index = ctx.index  # keep papers fetched during this run
        return res

    @staticmethod
    def _cited_ids(ans: dict) -> set[str]:
        inline = set(re.findall(r"\[([^\[\]\s]+#[^\[\]\s]+)\]", ans.get("answer", "")))
        for c in ans.get("conflicts", []) or []:
            inline |= set(re.findall(r"\[([^\[\]\s]+#[^\[\]\s]+)\]", c))
        return inline | set(ans.get("citations", []) or [])

    def _check(self, ans: dict, ctx: ToolContext, res: AgentResult) -> str | None:
        bad = sorted(c for c in self._cited_ids(ans) if c not in ctx.seen_chunks)
        if bad:
            return f"citation check failed: {bad} were never returned by a tool in this session"
        if ans.get("evidence_status") == "insufficient" and not any(s.tool in ("search_papers", "list_papers") for s in res.steps):
            return "abstain check failed: you must search the corpus before concluding the evidence is insufficient"
        if (ans.get("evidence_status") == "insufficient" and ctx.allow_network and not getattr(self, "_focus", None)
                and not any(s.tool in ("search_external_literature", "research_topic") for s in res.steps)):
            return ("external check failed: the corpus does not answer this, so search outside it with "
                    "search_external_literature; if a matching paper is found, add it with fetch_paper and answer from "
                    "it before concluding the evidence is insufficient")
        if ans.get("evidence_status") in ("supported", "partially_supported", "conflicting") and not self._cited_ids(ans):
            return "citation check failed: a supported answer needs at least one [chunk_id] citation"
        return None

    def _finalise(self, ans: dict, ctx: ToolContext, res: AgentResult):
        cited = self._cited_ids(ans)
        bad = sorted(c for c in cited if c not in ctx.seen_chunks)
        text = ans.get("answer", "")
        for b in bad:  # persisted after one retry: strip, flag, downgrade
            text = text.replace(f"[{b}]", "[citation removed]")
        res.answer = text
        res.citations = sorted(cited - set(bad))
        res.invalid_citations = bad
        res.evidence_status = ans.get("evidence_status", "insufficient")
        res.confidence = float(ans.get("confidence", 0) or 0)
        res.conflicts = ans.get("conflicts", []) or []
        if bad:
            res.confidence = min(res.confidence, 0.3)
            res.guardrail_events.append("invalid_citations_stripped")
