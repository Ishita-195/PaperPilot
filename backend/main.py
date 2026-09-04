"""FastAPI backend for the ML Research Assistant.

Endpoints
---------
GET  /api/health      liveness + model info
GET  /api/topics      knowledge-base topics grouped by category
POST /api/chat        ask a question, get answer + sources + faithfulness
WS   /api/chat/stream token-streamed answer over WebSocket
POST /api/upload      add PDF papers to the knowledge base (rebuilds index)
POST /api/evaluate    run the RAGAS-style suite and return live metrics
GET  /api/metrics     last computed metrics + before/after improvement
"""
from __future__ import annotations

import io
import json
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend import evaluation
from backend.agent import build_app, make_llm, FAITHFULNESS_THRESHOLD, LLM_MODEL
from backend.kb import DOCUMENTS, categories

ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = ROOT / "ragas_baseline.json"

app = FastAPI(title="ML Research Assistant API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Lazily built singletons (heavy: embedder, reranker, vector store) ----
STATE = {"app": None, "llm": None, "extra_docs": []}


def get_agent():
    if STATE["app"] is None:
        STATE["app"], _, _, _ = build_app(extra_docs=STATE["extra_docs"])
        STATE["llm"] = make_llm()
    return STATE["app"]


def rebuild_agent():
    STATE["app"], _, _, _ = build_app(extra_docs=STATE["extra_docs"])
    if STATE["llm"] is None:
        STATE["llm"] = make_llm()


# ================= MODELS =================
class ChatRequest(BaseModel):
    question: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    faithfulness: Optional[float]
    grounded: bool
    thread_id: str


# ================= ROUTES =================
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": LLM_MODEL,
        "kb_documents": len(DOCUMENTS) + len(STATE["extra_docs"]),
        "faithfulness_threshold": FAITHFULNESS_THRESHOLD,
    }


@app.get("/api/topics")
def topics():
    grouped = {c: [] for c in categories()}
    for d in DOCUMENTS:
        grouped[d["category"]].append(d["topic"])
    return grouped


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    agent = get_agent()
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"question": req.question, "messages": [], "eval_retries": 0}, config
    )
    faith = result.get("faithfulness")
    return ChatResponse(
        answer=result.get("answer", "Sorry, I could not generate a response."),
        sources=result.get("sources", []),
        faithfulness=faith,
        grounded=bool(faith is not None and faith >= FAITHFULNESS_THRESHOLD),
        thread_id=thread_id,
    )


@app.websocket("/api/chat/stream")
async def chat_stream(ws: WebSocket):
    await ws.accept()
    agent = get_agent()
    try:
        while True:
            payload = json.loads(await ws.receive_text())
            question = payload.get("question", "")
            thread_id = payload.get("thread_id") or str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}

            await ws.send_json({"type": "status", "value": "thinking"})
            result = agent.invoke(
                {"question": question, "messages": [], "eval_retries": 0}, config
            )
            answer = result.get("answer", "")

            # Stream the final answer word-by-word for a live typing effect.
            for word in answer.split(" "):
                await ws.send_json({"type": "token", "value": word + " "})
            await ws.send_json({
                "type": "done",
                "sources": result.get("sources", []),
                "faithfulness": result.get("faithfulness"),
                "thread_id": thread_id,
            })
    except WebSocketDisconnect:
        return


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    from pypdf import PdfReader

    added = []
    for i, f in enumerate(files):
        raw = await f.read()
        reader = PdfReader(io.BytesIO(raw))
        text = "".join((page.extract_text() or "") for page in reader.pages)
        if not text.strip():
            continue
        doc = {"id": f"pdf_{uuid.uuid4().hex[:8]}", "topic": f.filename,
               "category": "Uploaded", "text": text[:2000]}
        STATE["extra_docs"].append(doc)
        added.append(f.filename)

    if added:
        rebuild_agent()
    return {"added": added, "total_uploaded": len(STATE["extra_docs"])}


@app.post("/api/evaluate")
def evaluate():
    agent = get_agent()

    def ask_fn(q: str):
        config = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex[:6]}"}}
        return agent.invoke({"question": q, "messages": [], "eval_retries": 0}, config)

    results = evaluation.run_evaluation(ask_fn, STATE["llm"])

    # Compute improvement vs. the stored baseline (the "wow" number).
    if BASELINE_PATH.exists():
        try:
            baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
            base_cp = baseline.get("context_precision")
            if base_cp:
                delta = (results["context_precision"] - base_cp) / base_cp * 100
                results["context_precision_improvement_pct"] = round(delta, 1)
                results["baseline_context_precision"] = base_cp
        except Exception:
            pass

    BASELINE_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


@app.get("/api/metrics")
def metrics():
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return {"message": "No metrics yet. POST /api/evaluate to compute them."}
