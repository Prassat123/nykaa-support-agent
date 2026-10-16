"""
api.py - Tasks 11 & 12: FastAPI deployment + structured JSON-Lines logging.

Run:   uvicorn api:app --port 8080
Docs:  http://127.0.0.1:8080/docs   (Swagger UI)

Endpoints
  POST /ask           {query, session_id}  -> agent answer (validated schema)
  POST /add-document  {doc_id, text}       -> adds a policy doc to both Chroma collections
  GET  /health

Every request writes ONE line to logs/requests.jsonl with a trace_id and timing.
The logged text is PII-masked with the same function the agent uses, so a phone
number or card last-4 never reaches disk in the clear.
"""
import json
import time
from contextlib import asynccontextmanager
import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from agent import GraphTimeoutError, aask
from guardrails import mask_pii
from rag import add_document, ensure_index

@asynccontextmanager
async def lifespan(_app):
    ensure_index()          # build the Chroma index on startup if it does not exist
    yield


app = FastAPI(title="Nykaa Support Agent", version="1.0", lifespan=lifespan,
              description="LangGraph + RAG support agent for Nykaa (MOCK_LLM by default)")


# ------------------------------------------------------------- Pydantic models
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["What is the return window for Electronics?"])
    session_id: str = Field("default", examples=["customer-123"])


class OrderInfo(BaseModel):
    record_id: str
    status: str
    order_value_inr: int
    escalation_score: float
    escalate: bool


class Guardrails(BaseModel):
    pii_masked: list[str]
    injection_detected: bool
    grounded: bool
    note: str = ""


class AskResponse(BaseModel):
    trace_id: str
    session_id: str
    query: str
    route: str
    answer: str
    sources: list[str]
    order: OrderInfo | None
    guardrails: Guardrails
    nodes_executed: list[str]
    latency_ms: float


class AddDocRequest(BaseModel):
    doc_id: str = Field(..., min_length=1, examples=["13_gift_card_policy"])
    text: str = Field(..., min_length=10,
                      examples=["Nykaa gift cards are valid for 12 months from the date of purchase."])


class AddDocResponse(BaseModel):
    trace_id: str
    doc_id: str
    chunks_added: dict[str, int]
    latency_ms: float


# ------------------------------------------------------------- logging (Task 12)
def log_request(entry: dict) -> None:
    config.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with config.LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _base_log(trace_id: str, endpoint: str, text: str, start: float) -> dict:
    masked, pii = mask_pii(text)
    return {"trace_id": trace_id, "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint, "request_text": masked, "pii_masked": pii,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2)}


# ------------------------------------------------------------- endpoints (Task 11)
@app.get("/health")
def health():
    return {"status": "ok", "mock_llm": config.MOCK_LLM, "collection": config.DEPLOYED_COLLECTION}


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest):
    trace_id, start = str(uuid.uuid4()), time.perf_counter()
    try:
        resp = await aask(req.query, req.session_id)
    except GraphTimeoutError as e:
        log_request({**_base_log(trace_id, "/ask", req.query, start), "status": 504, "error": str(e)})
        raise HTTPException(status_code=504, detail=str(e))
    latency = round((time.perf_counter() - start) * 1000, 2)
    log_request({**_base_log(trace_id, "/ask", req.query, start), "session_id": req.session_id,
                 "status": 200, "route": resp["route"], "grounded": resp["guardrails"]["grounded"]})
    return AskResponse(trace_id=trace_id, latency_ms=latency, **resp)


@app.post("/add-document", response_model=AddDocResponse)
def add_document_endpoint(req: AddDocRequest):
    trace_id, start = str(uuid.uuid4()), time.perf_counter()
    counts = add_document(req.doc_id, req.text)
    log_request({**_base_log(trace_id, "/add-document", req.text, start),
                 "status": 200, "doc_id": req.doc_id, "chunks_added": counts})
    return AddDocResponse(trace_id=trace_id, doc_id=req.doc_id, chunks_added=counts,
                          latency_ms=round((time.perf_counter() - start) * 1000, 2))
