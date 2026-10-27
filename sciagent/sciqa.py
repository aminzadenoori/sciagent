"""SciQA: question answering over the ORKG scholarly knowledge graph via NL -> SPARQL.

Follows the evaluation setup of LLM-based NL-to-SPARQL work on SciQA
(e.g. Meloni et al.; Taffa & Usbeck, Scholarly-QALD-23):

* official split: train (example pool only) / valid / test;
* NO training and NO fine-tuning. Few-shot examples are the k most similar
  *train* questions under TF-IDF cosine similarity (a fixed, unlearned
  similarity; fitting the vectoriser vocabulary on train questions is the only
  "fit" and never sees test data);
* query-level metrics: BLEU-4, ROUGE-1/2/L, normalised exact match, literal F1;
* answer-level metrics: precision / recall / F1 of the answer set obtained by
  executing the generated query on an ORKG SPARQL endpoint, against the gold
  answers shipped with the dataset.

Generators
* `knn_template`  - non-LLM baseline: copy the nearest train query and replace
                    its string literals with the aligned spans of the test question.
* `ClaudeSparql`  - Claude with 0/1/3/5 retrieved examples.
"""
from __future__ import annotations

import difflib
import json
import math
import os
import re
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = "https://raw.githubusercontent.com/debayan/scholarly-QALD-challenge/main/2023/datasets/sciqa/SciQA-dataset"
SPLITS = ("train", "valid", "test")
ENDPOINT = os.environ.get("SCIQA_ENDPOINT", "https://orkg.org/triplestore")
PREFIXES = """PREFIX orkgr: <http://orkg.org/orkg/resource/>
PREFIX orkgc: <http://orkg.org/orkg/class/>
PREFIX orkgp: <http://orkg.org/orkg/predicate/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


# ------------------------------------------------------------------ data
@dataclass
class Item:
    id: str
    question: str
    sparql: str
    template_id: str | None
    handcrafted: bool
    query_type: str
    answers: list[str] | None = None


def download(root: Path) -> Path:
    for s in SPLITS:
        (root / s).mkdir(parents=True, exist_ok=True)
        for f in ("questions.json", "answers.json"):
            if not (root / s / f).exists():
                urllib.request.urlretrieve(f"{BASE}/{s}/{f}", root / s / f)
    return root


def answer_values(sparql_json: dict | None) -> list[str] | None:
    """Flatten SPARQL JSON results (or ASK boolean) into a list of string values."""
    if sparql_json is None:
        return None
    if "boolean" in sparql_json:
        return [str(sparql_json["boolean"]).lower()]
    out = []
    for b in sparql_json.get("results", {}).get("bindings", []):
        for v in b.values():
            out.append(_norm_value(v.get("value", "")))
    return out


def _norm_value(v: str) -> str:
    v = v.strip()
    try:  # 2857.3636e0 vs 2857.3636 -> same number
        f = float(v)
        return f"{f:.6g}"
    except ValueError:
        return v


def load_split(root: Path, split: str) -> list[Item]:
    qs = json.load(open(root / split / "questions.json"))["questions"]
    ans = {}
    p = root / split / "answers.json"
    if p.exists():
        for a in json.load(open(p))["answers"]:
            ans[a["id"]] = answer_values(a.get("answer"))
    return [Item(q["id"], q["question"]["string"], q["query"]["sparql"], q.get("template_id"),
                 not q.get("auto_generated", True), q.get("query_type", ""), ans.get(q["id"])) for q in qs]


# ------------------------------------------------------------------ example retrieval (no training)
class ExampleRetriever:
    def __init__(self, train: list[Item]):
        self.train = train
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)
        self.X = self.vec.fit_transform([t.question for t in train])

    def top_k(self, question: str, k: int) -> list[tuple[Item, float]]:
        if k <= 0:
            return []
        s = (self.X @ self.vec.transform([question]).T).toarray().ravel()
        idx = np.argsort(-s)[:k]
        return [(self.train[i], float(s[i])) for i in idx]


# ------------------------------------------------------------------ non-LLM baseline
LIT_RE = re.compile(r'"([^"]+)"')


def _tok(s):
    return re.findall(r"\w+(?:[-.'/]\w+)*|[^\w\s]", s)


def knn_template(question: str, retriever: ExampleRetriever) -> str:
    """Nearest train query; each literal that appears verbatim in the train question is
    replaced by the test-question span that aligns with it (difflib opcodes)."""
    ex, _ = retriever.top_k(question, 1)[0]
    q = ex.sparql
    a, b = _tok(ex.question), _tok(question)
    ops = difflib.SequenceMatcher(a=[x.lower() for x in a], b=[x.lower() for x in b], autojunk=False).get_opcodes()
    for lit in set(LIT_RE.findall(q)):
        lt = _tok(lit)
        n = len(lt)
        pos = next((i for i in range(len(a) - n + 1) if [x.lower() for x in a[i:i + n]] == [x.lower() for x in lt]), None)
        if pos is None:
            continue
        # map [pos, pos+n) in the train question to a span in the test question
        start = end = None
        for tag, i1, i2, j1, j2 in ops:
            if i1 <= pos < i2 or (i1 == i2 == pos):
                start = j1 + (pos - i1 if tag == "equal" else 0)
            if i1 < pos + n <= i2:
                end = j1 + (pos + n - i1 if tag == "equal" else j2 - j1)
        if start is None or end is None or end <= start:
            continue
        new = _detok(question, b[start:end])
        if new and new != lit:
            q = q.replace(f'"{lit}"', f'"{new}"')
    return q


def _detok(original: str, toks: list[str]) -> str:
    """Recover the original surface string for a token span."""
    if not toks:
        return ""
    pat = r"\s*".join(re.escape(t) for t in toks)
    m = re.search(pat, original)
    return m.group(0) if m else " ".join(toks)


# ------------------------------------------------------------------ Claude generator
SPARQL_SYSTEM = """You translate questions about scientific research into SPARQL for the Open Research Knowledge Graph (ORKG).
Prefixes orkgr:, orkgc:, orkgp:, rdfs:, rdf:, xsd: are predefined; do not declare them.
Typical patterns: papers -> contributions via orkgp:P31; contributions -> benchmarks via orkgp:HAS_BENCHMARK;
benchmarks -> orkgp:HAS_DATASET and orkgp:HAS_EVALUATION; evaluations -> orkgp:HAS_METRIC and orkgp:HAS_VALUE;
models via orkgp:HAS_MODEL; labels via rdfs:label matched with FILTER (str(?x_lbl) = "...").
Follow the style of the examples. Return ONLY the SPARQL query, no explanation, no code fences."""


class ClaudeSparql:
    def __init__(self, retriever: ExampleRetriever, client=None, model: str | None = None, k: int = 3):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.retriever, self.k = client, retriever, k
        self.model = model or os.environ.get("SCIAGENT_MODEL", "claude-sonnet-5-5")

    def prompt(self, question: str) -> str:
        shots = self.retriever.top_k(question, self.k)
        parts = [f"Question: {ex.question}\nSPARQL:\n{ex.sparql}\n" for ex, _ in shots]
        return ("Examples:\n\n" + "\n".join(parts) + "\n" if parts else "") + f"Question: {question}\nSPARQL:"

    def __call__(self, question: str) -> str:
        r = self.client.messages.create(model=self.model, max_tokens=800, temperature=0.0, system=SPARQL_SYSTEM,
                                        messages=[{"role": "user", "content": self.prompt(question)}])
        return clean_sparql("".join(getattr(b, "text", "") for b in r.content))


def clean_sparql(s: str) -> str:
    s = re.sub(r"^```(?:sparql)?\s*|\s*```$", "", s.strip(), flags=re.I)
    s = re.sub(r"(?im)^\s*PREFIX\s+\S+\s*<[^>]+>\s*$", "", s)  # prefixes are added at execution
    return s.strip()


# ------------------------------------------------------------------ execution
def execute(sparql: str, endpoint: str = ENDPOINT, timeout: int = 30) -> list[str] | None:
    """Run on the ORKG endpoint; None = execution failure (syntax error, timeout, network)."""
    url = endpoint + "?" + urllib.parse.urlencode({"query": PREFIXES + sparql, "format": "json"})
    req = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json", "User-Agent": "sciagent/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return answer_values(json.loads(r.read()))
    except Exception:
        return None


# ------------------------------------------------------------------ metrics
def sparql_tokens(q: str) -> list[str]:
    return re.findall(r'"[^"]*"|<[^>]*>|\?\w+|[\w:]+|[{}().;,=<>!*]', q)


def normalise_query(q: str) -> str:
    """Canonical form for exact match: whitespace-insensitive, variables renamed in order of appearance."""
    toks = sparql_tokens(q)
    names = {}
    out = []
    for t in toks:
        if t.startswith("?"):
            names.setdefault(t, f"?v{len(names)}")
            t = names[t]
        out.append(t.lower() if not t.startswith('"') else t)
    return " ".join(out)


def bleu4(refs: list[list[str]], hyps: list[list[str]]) -> float:
    """Corpus BLEU-4 with add-1 smoothing for n>1 (Lin & Och 2004)."""
    num, den = [0] * 4, [0] * 4
    rl = hl = 0
    for r, h in zip(refs, hyps):
        rl, hl = rl + len(r), hl + len(h)
        for n in range(1, 5):
            rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
            num[n - 1] += sum(min(c, rc[g]) for g, c in hc.items())
            den[n - 1] += max(0, len(h) - n + 1)
    if hl == 0:
        return 0.0
    if num[0] == 0 or den[0] == 0:
        return 0.0
    logp = (math.log(num[0] / den[0]) + sum(math.log((num[i] + 1) / (den[i] + 1)) for i in range(1, 4))) / 4
    bp = 1.0 if hl > rl else math.exp(1 - rl / hl)
    return bp * math.exp(logp)


def _lcs(a, b):
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b):
            cur = dp[j + 1]
            dp[j + 1] = prev + 1 if x == y else max(dp[j + 1], dp[j])
            prev = cur
    return dp[-1]


def rouge(ref: list[str], hyp: list[str]) -> dict:
    def f(p, r):
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)
    out = {}
    for n in (1, 2):
        rc = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        hc = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
        ov = sum((rc & hc).values())
        out[f"rouge{n}"] = f(ov / max(1, sum(hc.values())), ov / max(1, sum(rc.values())))
    l = _lcs(ref, hyp)
    out["rougeL"] = f(l / max(1, len(hyp)), l / max(1, len(ref)))
    return out


def set_prf(gold: list[str] | None, pred: list[str] | None) -> tuple[float, float, float]:
    """Answer-set P/R/F1 (Scholarly-QALD style). Empty gold & empty pred = perfect."""
    g, p = set(gold or []), set(pred or [])
    if not g and not p:
        return 1.0, 1.0, 1.0
    if not g or not p:
        return 0.0, 0.0, 0.0
    tp = len(g & p)
    pr, rc = tp / len(p), tp / len(g)
    return pr, rc, (0.0 if tp == 0 else 2 * pr * rc / (pr + rc))


def literal_f1(gold_q: str, pred_q: str) -> float:
    """Did the query mention the right entities/values? F1 over quoted literals and orkgr: IRIs."""
    def lits(q):
        return set(LIT_RE.findall(q)) | set(re.findall(r"orkgr:R\d+", q))
    g, p = lits(gold_q), lits(pred_q)
    return set_prf(list(g), list(p))[2]


def evaluate(items: list[Item], preds: dict[str, str], executed: dict[str, list[str] | None] | None = None) -> dict:
    """Aggregate metrics, overall and for handcrafted (HQ) vs template-generated (AQ) questions."""
    def block(sub):
        if not sub:
            return None
        refs = [sparql_tokens(i.sparql) for i in sub]
        hyps = [sparql_tokens(preds.get(i.id, "")) for i in sub]
        rg = [rouge(r, h) for r, h in zip(refs, hyps)]
        res = {"n": len(sub), "BLEU-4": round(100 * bleu4(refs, hyps), 2),
               **{k.upper().replace("ROUGE", "ROUGE-"): round(100 * float(np.mean([x[k] for x in rg])), 2) for k in ("rouge1", "rouge2", "rougeL")},
               "exact_match": round(100 * float(np.mean([normalise_query(i.sparql) == normalise_query(preds.get(i.id, "")) for i in sub])), 2),
               "literal_F1": round(100 * float(np.mean([literal_f1(i.sparql, preds.get(i.id, "")) for i in sub])), 2)}
        if executed is not None:
            ev = [i for i in sub if i.answers is not None]
            prf = [set_prf(i.answers, executed.get(i.id)) for i in ev]
            if prf:
                res.update({"answer_P": round(100 * float(np.mean([x[0] for x in prf])), 2),
                            "answer_R": round(100 * float(np.mean([x[1] for x in prf])), 2),
                            "answer_F1": round(100 * float(np.mean([x[2] for x in prf])), 2),
                            "exec_fail_%": round(100 * float(np.mean([executed.get(i.id) is None for i in ev])), 2)})
        return res
    return {"all": block(items), "handcrafted": block([i for i in items if i.handcrafted]),
            "template_generated": block([i for i in items if not i.handcrafted])}


def template_hit_at_k(items: list[Item], retriever: ExampleRetriever, ks=(1, 3, 5)) -> dict:
    """Example-retrieval quality: share of test questions with a same-template train example in the top-k."""
    sub = [i for i in items if i.template_id]
    out = {}
    for k in ks:
        out[f"template_hit@{k}"] = round(100 * float(np.mean(
            [any(ex.template_id == i.template_id for ex, _ in retriever.top_k(i.question, k)) for i in sub])), 2)
    out["n"] = len(sub)
    return out
