"""
agent.py - Tasks 7, 8, 9, 10 (+ hooks for 15 & 16): the LangGraph support agent.

Graph (7 nodes, 1 conditional edge):

  START -> input_guard -> classify_intent --(conditional)--> retrieve_policy --+
                                          |                                    |
                                          +--> lookup_order -------------------+--> generate_answer
                                          |                                    |        |
                                          +--> (blocked / clarify) ------------+        v
                                                                                   output_guard -> finalize -> END

input_guard     : masks PII + detects prompt injection             (Task 10, input side)
classify_intent : reads JSON memory, decides the route              (Task 7 + Task 8)
retrieve_policy : RAG retrieval from ChromaDB  [per-node timeout]   (Task 3-5, Task 16)
lookup_order    : check_order_status tool      [retry policy]       (Task 6, Task 16)
generate_answer : grounded answer / order summary                   (Task 4)
output_guard    : groundedness check, refuses unsupported answers   (Task 10, output side)
finalize        : JSON-schema validation + save memory              (Task 9 + Task 8)
"""
import asyncio
import operator
import re
import time
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, TimeoutPolicy

import config
import memory
from guardrails import detect_injection, is_grounded, mask_pii
from rag import FALLBACK_ANSWER, grounded_answer, retrieve
from llm import generate_answer as llm_generate
from schema import validate_response
from tools import check_order_status

ORDER_ID_RE = re.compile(r"\bNYK-?\d{4}\b", re.IGNORECASE)
ORDER_WORDS = re.compile(r"\b(status|track|tracking|where|delayed|late|shipped|delivered|"
                         r"arrive|escalat\w*|value|cost)\b", re.IGNORECASE)
FOLLOW_UP_WORDS = re.compile(r"\b(it|that order|this order|my order|same order|the order)\b", re.IGNORECASE)

EXECUTION_LOG: list[str] = []          # every node run in THIS process (used by Task 15 demo)
_TRANSIENT_CALLS = {"count": 0, "t0": 0.0}   # counter for the simulated transient failure (Task 16)
VERBOSE = {"on": False}


class TransientLookupError(ConnectionError):
    """Simulated temporary failure of the order database (Task 16 demo)."""


class AgentState(TypedDict, total=False):
    session_id: str
    raw_query: str
    query: str
    pii_found: list[str]
    injection: bool
    route: str
    order_id: str | None
    order_from_memory: bool
    order: dict | None
    chunks: list[dict]
    answer: str
    sources: list[str]
    is_fallback: bool
    grounded: bool
    guard_note: str
    hallucinate: bool                  # demo flag: simulate a model that invents a claim
    simulate: dict[str, Any]           # demo flags for Task 16
    nodes_executed: Annotated[list[str], operator.add]
    response: dict


def _log(name: str):
    EXECUTION_LOG.append(name)
    if VERBOSE["on"]:
        print(f"    [EXECUTING NODE] {name}")
    return [name]


async def _maybe_slow(state: AgentState):
    delay = state.get("simulate", {}).get("slow_every_node_seconds", 0)
    if delay:
        await asyncio.sleep(delay)


# ------------------------------------------------------------------ nodes
async def input_guard(state: AgentState) -> dict:
    await _maybe_slow(state)
    masked, found = mask_pii(state["raw_query"])
    return {"query": masked, "raw_query": masked,       # raw text is replaced, never kept
            "pii_found": found, "injection": detect_injection(masked),
            "nodes_executed": _log("input_guard")}


async def classify_intent(state: AgentState) -> dict:
    await _maybe_slow(state)
    q = state["query"]
    session = memory.load_session(state["session_id"])
    match = ORDER_ID_RE.search(q)
    if state.get("injection"):
        route, order_id = "blocked", None
    elif match:
        route, order_id = "order_lookup", match.group(0).upper().replace("NYK", "NYK-").replace("--", "-")
    elif ORDER_WORDS.search(q) and FOLLOW_UP_WORDS.search(q):
        order_id = session.get("last_order_id")              # resolved from JSON memory
        route = "order_lookup" if order_id else "clarify"
    else:
        route, order_id = "rag", None
    return {"route": route, "order_id": order_id,
            "order_from_memory": route == "order_lookup" and not match,
            "nodes_executed": _log("classify_intent")}


def route_after_classify(state: AgentState) -> str:
    """The conditional edge: send the query to the right tool."""
    return {"rag": "retrieve_policy", "order_lookup": "lookup_order"}.get(state["route"], "generate_answer")


async def retrieve_policy(state: AgentState) -> dict:
    slow = state.get("simulate", {}).get("slow_retrieve_seconds", 0)
    if slow:
        await asyncio.sleep(slow)                            # simulated slow vector DB
    await _maybe_slow(state)
    chunks = retrieve(state["query"], config.DEPLOYED_COLLECTION)
    return {"chunks": chunks, "nodes_executed": _log("retrieve_policy")}


async def lookup_order(state: AgentState) -> dict:
    await _maybe_slow(state)
    failures = state.get("simulate", {}).get("transient_failures", 0)
    if failures and _TRANSIENT_CALLS["count"] == 0:
        _TRANSIENT_CALLS["t0"] = time.perf_counter()
    elapsed = time.perf_counter() - _TRANSIENT_CALLS["t0"]
    if _TRANSIENT_CALLS["count"] < failures:
        _TRANSIENT_CALLS["count"] += 1
        msg = f"order DB temporarily unavailable (simulated failure {_TRANSIENT_CALLS['count']}/{failures})"
        if VERBOSE["on"]:
            print(f"    [lookup_order] t={elapsed:5.2f}s attempt {_TRANSIENT_CALLS['count']} FAILED: {msg}")
        raise TransientLookupError(msg)
    if failures and VERBOSE["on"]:
        print(f"    [lookup_order] t={elapsed:5.2f}s attempt {_TRANSIENT_CALLS['count'] + 1} SUCCEEDED")
    return {"order": check_order_status(state["order_id"]), "nodes_executed": _log("lookup_order")}


async def generate_answer(state: AgentState) -> dict:
    await _maybe_slow(state)
    route = state["route"]
    out: dict = {"nodes_executed": _log("generate_answer"), "sources": [], "is_fallback": False}
    if route == "blocked":
        out["answer"] = ("Sorry, I can't follow that request. I can only help with Nykaa "
                         "orders and policies.")
    elif route == "clarify":
        out["answer"] = "Please share your order ID (for example NYK-1001) so I can check it."
    elif route == "order_lookup":
        o = state["order"]
        if not o["found"]:
            out["answer"] = o["message"]
        else:
            action = ("This order needs escalation to Level 2 support."
                      if o["escalate"] else "No escalation needed.")
            prefix = (f"(From our conversation, you mean order {o['record_id']}.) "
                      if state.get("order_from_memory") else "")
            out["answer"] = prefix + (f"Order {o['record_id']} ({o['category']}) is currently {o['status']}. "
                             f"Order value: INR {o['order_value_inr']}. "
                             f"Delayed shipment: {'Yes' if o['delayed_shipment'] else 'No'}. "
                             f"Escalation score: {o['escalation_score']} "
                             f"(threshold {o['escalation_threshold']}). {action}")
            out["sources"] = ["orders_dataset"]
    else:  # rag
        result = grounded_answer(state["query"], chunks=state["chunks"])
        answer = result["answer"]
        if state.get("hallucinate") and not result["fallback"]:
            answer = llm_generate(state["query"], result["chunks"], hallucinate=True)
        out.update(answer=answer, sources=result["sources"], is_fallback=result["fallback"],
                   chunks=result["chunks"])
    return out


async def output_guard(state: AgentState) -> dict:
    await _maybe_slow(state)
    out = {"nodes_executed": _log("output_guard")}
    if state["route"] != "rag":
        out.update(grounded=True, guard_note="answer built from tool data / fixed template")
    elif state.get("is_fallback"):
        top1 = state["chunks"][0]["similarity"] if state.get("chunks") else 0.0
        out.update(grounded=False, guard_note=(
            f"refused: retrieved context does not support the question "
            f"(top-1 similarity {top1} < threshold {config.load_threshold()})"))
    else:
        ok, reason = is_grounded(state["answer"], state["chunks"], state["query"])
        out.update(grounded=ok, guard_note=reason)
        if not ok:
            out.update(answer=FALLBACK_ANSWER, sources=[])
    return out


async def finalize(state: AgentState) -> dict:
    await _maybe_slow(state)
    nodes = state.get("nodes_executed", []) + ["finalize"]
    _log("finalize")
    order = state.get("order")
    response = {
        "session_id": state["session_id"],
        "query": state["query"],
        "route": state["route"],
        "answer": state["answer"],
        "sources": state.get("sources", []),
        "order": ({k: order[k] for k in ("record_id", "status", "order_value_inr",
                                         "escalation_score", "escalate")}
                  if order and order.get("found") else None),
        "guardrails": {"pii_masked": state.get("pii_found", []),
                       "injection_detected": bool(state.get("injection")),
                       "grounded": bool(state.get("grounded")),
                       "note": state.get("guard_note", "")},
        "nodes_executed": nodes,
    }
    validate_response(response)                              # Task 9: every response checked
    found_id = order["record_id"] if order and order.get("found") else None
    memory.save_turn(state["session_id"], state["query"], state["answer"], state["route"], found_id)
    return {"response": response, "nodes_executed": ["finalize"]}


# ------------------------------------------------------------------ graph
RETRY_POLICY = RetryPolicy(
    max_attempts=config.RETRY_MAX_ATTEMPTS,
    initial_interval=config.RETRY_INITIAL_INTERVAL,
    backoff_factor=config.RETRY_BACKOFF_FACTOR,
    max_interval=config.RETRY_MAX_INTERVAL,
    jitter=config.RETRY_JITTER,
    retry_on=TransientLookupError,
)


def build_graph(checkpointer=None, interrupt_before=None):
    g = StateGraph(AgentState)
    g.add_node("input_guard", input_guard)
    g.add_node("classify_intent", classify_intent)
    g.add_node("retrieve_policy", retrieve_policy,
               timeout=TimeoutPolicy(run_timeout=config.NODE_TIMEOUT_SECONDS))
    g.add_node("lookup_order", lookup_order, retry_policy=RETRY_POLICY)
    g.add_node("generate_answer", generate_answer)
    g.add_node("output_guard", output_guard)
    g.add_node("finalize", finalize)

    g.add_edge(START, "input_guard")
    g.add_edge("input_guard", "classify_intent")
    g.add_conditional_edges("classify_intent", route_after_classify,
                            ["retrieve_policy", "lookup_order", "generate_answer"])
    g.add_edge("retrieve_policy", "generate_answer")
    g.add_edge("lookup_order", "generate_answer")
    g.add_edge("generate_answer", "output_guard")
    g.add_edge("output_guard", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


GRAPH = build_graph()


class GraphTimeoutError(TimeoutError):
    """Raised when one whole request exceeds GLOBAL_TIMEOUT_SECONDS."""


async def aask(query: str, session_id: str = "default", *, hallucinate: bool = False,
               simulate: dict | None = None, global_timeout: float = config.GLOBAL_TIMEOUT_SECONDS) -> dict:
    """Run one query through the graph with a global (whole-graph) timeout."""
    _TRANSIENT_CALLS["count"] = 0
    state = {"session_id": session_id, "raw_query": query, "hallucinate": hallucinate,
             "simulate": simulate or {}, "nodes_executed": []}
    try:
        result = await asyncio.wait_for(GRAPH.ainvoke(state), timeout=global_timeout)
    except asyncio.TimeoutError as exc:
        raise GraphTimeoutError(f"whole graph exceeded the global timeout of {global_timeout}s") from exc
    return result["response"]


def ask(query: str, session_id: str = "default", **kwargs) -> dict:
    return asyncio.run(aask(query, session_id, **kwargs))


if __name__ == "__main__":
    import json
    from rag import ensure_index
    ensure_index()
    print(json.dumps(ask("How many days do I have to return shoes?", "cli"), indent=2))
