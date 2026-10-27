"""Ingestion (text / tables / figures, degraded inputs) and hybrid retrieval."""
import json

import pytest


def chunks_of(built, pid, kind=None):
    return [c for c in built.index.chunks if c["paper_id"] == pid and (kind is None or c["kind"] == kind)]


# ------------------------------------------------------------------ text
def test_sections_detected_and_prefixed(built):
    secs = {c["section"] for c in chunks_of(built, "toy:good", "text")}
    assert {"Abstract", "Method", "Results", "Conclusion"} <= secs
    for c in chunks_of(built, "toy:good", "text"):
        assert c["text"].startswith("[Toy embeddings for knowledge graphs > ")  # contextual header


def test_ligatures_normalised(built):
    text = " ".join(c["text"] for c in chunks_of(built, "toy:good"))
    assert "ﬁ" not in text and "configuration" in text  # "ﬁ" ligature -> "fi"


def test_chunking_overlap():
    from sciagent.ingest import chunk_words
    words = [f"w{i}" for i in range(300)]
    parts = list(chunk_words(words, size=100, overlap=20))
    assert parts[0].split()[-20:] == parts[1].split()[:20]
    assert set(" ".join(parts).split()) == set(words)  # nothing lost


# ------------------------------------------------------------------ multimodal
def test_table_extracted_with_exact_decimals(built):
    tabs = chunks_of(built, "toy:good", "table")
    assert len(tabs) == 1
    rows = tabs[0]["meta"]["rows"]
    assert rows[0] == ["Model", "MRR", "Hits@10"]
    assert ["ToyE", "0.427", "0.512"] in rows  # cell-wise extraction keeps "0.427" intact
    assert "Table 1. Results on ToyKG." in tabs[0]["text"]


def test_table_text_not_duplicated_into_body_or_caption(built):
    body = " ".join(c["text"] for c in chunks_of(built, "toy:good", "text"))
    assert "0.512" not in body


def test_figure_extracted_with_caption_and_image(built):
    from pathlib import Path
    figs = chunks_of(built, "toy:good", "figure")
    assert len(figs) == 1
    assert figs[0]["meta"]["caption"].startswith("Figure 1. Score of system A")
    assert Path(figs[0]["meta"]["image_path"]).exists()


# ------------------------------------------------------------------ degraded inputs
def test_corrupt_pdf_falls_back_to_abstract(built):
    assert built.report["toy:corrupt"].get("abstract_only")
    (c,) = chunks_of(built, "toy:corrupt")
    assert "negative sampling" in c["text"]


def test_image_only_pdf_flagged_for_ocr(built):
    assert built.report["toy:scan"]["needs_ocr"] is True
    assert built.report["toy:scan"].get("abstract_only")


def test_missing_pdf_falls_back_to_abstract(built):
    assert built.report["toy:missing"].get("missing_pdf") and built.report["toy:missing"].get("abstract_only")


def test_duplicate_record_prefers_full_pdf_over_card(built):
    assert not any("card text that should lose" in c["text"] for c in built.index.chunks)
    assert built.index.papers["toy:good"]["title"] == "Toy embeddings for knowledge graphs"


def test_card_records_are_indexed(built):
    assert {c["section"] for c in chunks_of(built, "toy:card")} == {"Method", "Results"}


# ------------------------------------------------------------------ retrieval
@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_exact_scientific_tokens_found(built, mode):
    hits = built.index.search("CardNet exact match CardQA", k=3, mode=mode)
    assert hits[0].paper_id == "toy:card"


def test_dense_handles_paraphrase(built):
    hits = built.index.search("aligning graph embeddings with language models", k=3, mode="dense")
    assert "toy:card" in [h.paper_id for h in hits]


def test_kind_and_paper_filters(built):
    assert all(h.kind == "figure" for h in built.index.search("score of system", kinds=["figure"]))
    assert all(h.paper_id == "toy:card" for h in built.index.search("results", paper_ids=["toy:card"]))
    assert built.index.search("results", paper_ids=["does-not-exist"]) == []


def test_per_paper_cap(built):
    hits = built.index.search("ToyE ToyKG MRR rotations", k=10, per_paper=2)
    assert max(sum(h.paper_id == p for h in hits) for p in {h.paper_id for h in hits}) <= 2


def test_empty_query(built):
    assert built.index.search("   ") == []


def test_index_roundtrip(built):
    from sciagent.index import HybridIndex
    again = HybridIndex.load(built.dir)
    assert [h.chunk_id for h in again.search("CardNet")] == [h.chunk_id for h in built.index.search("CardNet")]
