"""LLM-as-judge: compare the system's answer with a golden (reference) answer.

For each question the judge sees the question, the golden answer, the kind of answer
expected, and the system's answer. It returns a verdict from a fixed rubric plus the
key points matched, missing and contradicted, so every verdict can be audited.

Verdicts and scores
  correct              1.0  all key points of the golden answer, nothing contradicting it
  partially_correct    0.5  some key points, nothing contradicting them
  incorrect            0.0  contradicts the golden answer or misses its main point
  correct_abstention   1.0  golden answer is "not available" and the system says so
  wrong_abstention     0.0  system says "not available" but the golden answer has the answer
  hallucinated         0.0  golden answer is "not available" but the system gives specific values

Bias controls: temperature 0, a rubric with definitions, the golden answer shown as the
reference (not the system's own sources), and an option to use a different model as judge
than the one answering (`--judge-model`) to reduce self-preference. The judge's own
reliability is measured on a hand-labelled calibration set (evals/judge_calibration.jsonl).
"""
from __future__ import annotations

import json
import re

VERDICTS = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0, "correct_abstention": 1.0,
            "wrong_abstention": 0.0, "hallucinated": 0.0}

JUDGE_SYSTEM = """You grade answers of a scientific question-answering system against a golden reference answer.

Rules:
- Compare MEANING, not wording. Paraphrases, synonyms and reasonable rounding of numbers are fine \
(e.g. 0.336 vs 33.6% vs 0.3359). A different number is a contradiction.
- Extra information is fine unless it contradicts the golden answer.
- Ignore citation markers like [paper#t3]; they are not part of the meaning.
- answer_type "fact": the golden answer lists the key points. correct = all key points present and nothing \
contradicts them; partially_correct = some key points present, none contradicted; incorrect = contradicts a key point \
or misses the main point. If the system says the information is not available although the golden answer has it: \
wrong_abstention.
- answer_type "abstain": the correct behaviour is to say the information is not available. correct_abstention = the \
system says so and gives no invented values; hallucinated = it states specific values or claims as if found.
- answer_type "open": the golden answer lists criteria. correct = all criteria met; partially_correct = most met; \
incorrect = the main criteria are not met or unsupported numbers are presented as fact.
- If the golden answer says papers disagree, an answer that presents only one side as settled misses a key point.

Reply with JSON only:
{"verdict": "correct|partially_correct|incorrect|correct_abstention|wrong_abstention|hallucinated",
 "matched_points": ["..."], "missing_points": ["..."], "contradictions": ["..."], "explanation": "one or two sentences"}"""

JUDGE_USER = """Question: {question}

answer_type: {answer_type}

Golden answer:
{reference}

System answer (evidence status reported by the system: {status}):
{answer}"""


def strip_citations(text: str) -> str:
    return re.sub(r"\s*\[[^\[\]\s]+#[^\[\]\s]+\]", "", text or "")


def parse_verdict(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return {"verdict": "unparseable", "score": None, "raw": (text or "")[:300]}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"verdict": "unparseable", "score": None, "raw": m.group(0)[:300]}
    v = str(d.get("verdict", "")).strip().lower().replace(" ", "_")
    d["verdict"] = v if v in VERDICTS else "unparseable"
    d["score"] = VERDICTS.get(d["verdict"])
    return d


def judge(client, model: str, question: str, reference: str, answer: str, answer_type: str = "fact",
          status: str | None = None) -> dict:
    msg = JUDGE_USER.format(question=question, answer_type=answer_type, reference=reference,
                            answer=strip_citations(answer).strip() or "(empty answer)", status=status or "not given")
    resp = client.messages.create(model=model, max_tokens=600, temperature=0.0, system=JUDGE_SYSTEM,
                                  messages=[{"role": "user", "content": msg}])
    return parse_verdict("".join(getattr(b, "text", "") for b in resp.content))


def summarize(judgements: list[dict]) -> dict:
    ok = [j for j in judgements if j.get("score") is not None]
    counts = {v: sum(j.get("verdict") == v for j in judgements) for v in list(VERDICTS) + ["unparseable"]}
    return {"n": len(judgements), "mean_score": round(sum(j["score"] for j in ok) / len(ok), 3) if ok else None,
            "verdicts": {k: v for k, v in counts.items() if v}}
