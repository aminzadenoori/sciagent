import json

from sciagent.tools import ToolContext, call_tool, tool_specs


def ctx(built, client=None):
    return ToolContext(built.index, built.dir, client=client, model="m", allow_network=False)


def test_tool_schemas_are_valid():
    specs = tool_specs()
    names = [s["name"] for s in specs]
    assert len(names) == len(set(names)) and names[-1] == "submit_answer"
    for s in specs:
        assert s["description"] and s["input_schema"]["type"] == "object"
        for r in s["input_schema"].get("required", []):
            assert r in s["input_schema"]["properties"]


def test_search_registers_seen_chunks(built):
    c = ctx(built)
    out = json.loads(call_tool(c, "search_papers", {"query": "CardNet", "k": 3}))
    assert out["n"] >= 1 and {r["chunk_id"] for r in out["results"]} <= c.seen_chunks
    assert all("paper_title" in r for r in out["results"])


def test_search_with_no_results_says_so(built):
    out = json.loads(call_tool(ctx(built), "search_papers", {"query": "x", "paper_ids": ["none"]}))
    assert out["n"] == 0 and "No matching evidence" in out["note"]


def test_read_chunks_reports_missing(built):
    out = json.loads(call_tool(ctx(built), "read_chunks", {"chunk_ids": ["toy:card#t0", "bogus#1"]}))
    assert [c["chunk_id"] for c in out["chunks"]] == ["toy:card#t0"] and out["missing"] == ["bogus#1"]


def test_list_papers_filters_and_is_citable(built):
    c = ctx(built)
    out = json.loads(call_tool(c, "list_papers", {"author": "lovelace"}))
    assert {p["paper_id"] for p in out["papers"]} == {"toy:good", "toy:card"}
    assert "toy:card#meta" in c.seen_chunks


def test_outline_lists_figures_and_tables(built):
    out = json.loads(call_tool(ctx(built), "get_paper_outline", {"paper_id": "toy:good"}))
    kinds = {p["kind"] for p in out["parts"]}
    assert {"text", "table", "figure"} <= kinds


def test_inspect_figure_without_vision_returns_caption(built):
    fig = built.index.search("system", kinds=["figure"])[0].chunk_id
    out = json.loads(call_tool(ctx(built), "inspect_figure", {"chunk_id": fig, "question": "values?"}))
    assert out["caption"].startswith("Figure 1") and "caption only" in out["note"]


def test_inspect_figure_sends_image_to_vision_model(built, fake):
    fig = built.index.search("system", kinds=["figure"])[0].chunk_id
    client = fake.Client([fake.response(fake.text("Bar A is about 90, bar B about 10."))])
    out = json.loads(call_tool(ctx(built, client), "inspect_figure", {"chunk_id": fig, "question": "values?"}))
    assert "90" in out["vision_reading"]
    content = client.calls[0]["messages"][0]["content"]
    assert content[0]["type"] == "image" and content[0]["source"]["media_type"] == "image/png"


def test_inspect_figure_rejects_non_figure(built):
    out = json.loads(call_tool(ctx(built), "inspect_figure", {"chunk_id": "toy:card#t0", "question": "?"}))
    assert "not a figure" in out["error"]


def test_run_python(built):
    out = json.loads(call_tool(ctx(built), "run_python", {"code": "import numpy as np; print(np.mean([0.396, 0.243]))"}))
    assert out["returncode"] == 0 and out["stdout"].strip() == "0.3195"
    out = json.loads(call_tool(ctx(built), "run_python", {"code": "import time; time.sleep(5)", "timeout": 1}))
    assert "timed out" in out["error"]
    out = json.loads(call_tool(ctx(built), "run_python", {"code": "1/0"}))
    assert out["returncode"] != 0 and "ZeroDivisionError" in out["stderr"]


def test_offline_tools_fail_gracefully(built):
    c = ctx(built)
    assert "network disabled" in json.loads(call_tool(c, "search_external_literature", {"query": "x"}))["error"]
    assert "unknown dataset" in json.loads(call_tool(c, "kg_leakage_analysis", {"dataset": "YAGO3-10"}))["error"]
    assert "unavailable" in json.loads(call_tool(c, "query_scholarly_kg", {"question": "x"}))["error"]
