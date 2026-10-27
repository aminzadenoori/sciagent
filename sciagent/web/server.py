"""Small local web interface: python -m sciagent serve  ->  http://localhost:8000

Standard library only (http.server). The agent run is streamed to the browser
with Server-Sent Events so each tool call appears as it happens. Without an
ANTHROPIC_API_KEY the page still works in search-only mode.
"""
from __future__ import annotations

import json
import os
import re
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent


class App:
    def __init__(self, data_dir: Path, model: str):
        from ..index import HybridIndex
        self.data_dir = data_dir
        self.model = model
        self.index = HybridIndex.load(data_dir / "index")
        self.lock = threading.Lock()

    def has_key(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def source(self, cid: str) -> dict:
        pid = cid.split("#")[0]
        p = self.index.papers.get(pid, {})
        c = self.index.by_id.get(cid)
        base = {"id": cid, "paper_id": pid, "paper_title": p.get("title"), "year": p.get("year"),
                "authors": p.get("authors", [])[:6], "url": p.get("url"), "is_card": bool(p.get("sections"))}
        if cid.endswith("#meta"):
            return {**base, "kind": "metadata", "section": "Bibliographic record", "text": p.get("title", "")}
        if cid.startswith("orkg:"):
            return {**base, "kind": "kg", "section": "ORKG query result", "text": "Result of a SPARQL query on ORKG."}
        if not c:
            return {**base, "kind": "unknown", "section": "", "text": ""}
        return {**base, "kind": c["kind"], "section": c["section"], "text": re.sub(r"^\[[^\]]*\]\s*", "", c["text"]),
                "caption": c.get("meta", {}).get("caption"), "has_image": bool(c.get("meta", {}).get("image_path"))}


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    body = (HERE / "index.html").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif u.path == "/api/status":
                    kinds = {}
                    for c in app.index.chunks:
                        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
                    self._json({"papers": len(app.index.papers), "chunks": kinds, "model": app.model,
                                "has_key": app.has_key()})
                elif u.path == "/api/papers":
                    rows = sorted(({"id": p["id"], "title": p["title"], "year": p.get("year"),
                                    "authors": p.get("authors", [])[:4]} for p in app.index.papers.values()),
                                  key=lambda r: -(r["year"] or 0))
                    self._json(rows)
                elif u.path == "/api/search":
                    hits = app.index.search(q.get("q", ""), k=int(q.get("k", 8)))
                    self._json([{**app.source(h.chunk_id), "score": round(h.score, 4)} for h in hits])
                elif u.path == "/api/source":
                    self._json(app.source(q.get("id", "")))
                elif u.path == "/api/figure":
                    c = app.index.by_id.get(q.get("id", ""))
                    path = Path(c["meta"]["image_path"]) if c and c.get("meta", {}).get("image_path") else None
                    if not path or not path.exists():
                        return self._json({"error": "no image"}, 404)
                    body = path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif u.path == "/api/ask":
                    self._ask(q.get("q", "").strip())
                else:
                    self._json({"error": "not found"}, 404)
            except BrokenPipeError:
                pass
            except Exception as e:
                traceback.print_exc()
                try:
                    self._json({"error": str(e)}, 500)
                except Exception:
                    pass

        def _ask(self, question: str):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def send(ev):
                self.wfile.write(f"data: {json.dumps(ev, default=str)}\n\n".encode())
                self.wfile.flush()

            if not question:
                return send({"type": "error", "message": "Type a question first."})
            if not app.has_key():
                return send({"type": "error", "message": "ANTHROPIC_API_KEY is not set, so only search is available."})
            from ..agent import ResearchAgent
            agent = ResearchAgent(app.index, app.data_dir, model=app.model)
            send({"type": "start"})
            r = agent.run(question, on_event=send)
            d = r.to_dict()
            d["sources"] = {cid: app.source(cid) for cid in r.citations}
            send({"type": "final", "result": d})

    return H


def serve(data_dir: Path, port: int = 8000, model: str | None = None):
    app = App(data_dir, model or os.environ.get("SCIAGENT_MODEL", "claude-sonnet-5-5"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    mode = "agent" if app.has_key() else "search-only (set ANTHROPIC_API_KEY for answers)"
    print(f"sciagent web UI on http://localhost:{port}  [{mode}]")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
