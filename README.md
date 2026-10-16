# Nykaa Support Agent — Final Capstone

> **Track completed: Nykaa (E-commerce & Retail).**
> Everything runs under **`MOCK_LLM`** with **zero API keys** and no paid accounts.

A production-minded customer-support agent for Nykaa that answers policy questions from a
knowledge base (RAG over ChromaDB), looks up order status from a validated dataset, remembers the
conversation, is guarded against PII leaks / prompt injection / ungrounded answers, survives
interruptions and transient failures, exposes its lookup tool over MCP, and is deployed behind FastAPI.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py            # runs every task and writes transcripts/ (first run downloads the free embedding model)
```

Other entry points:

```bash
uvicorn api:app --port 8080          # FastAPI -> open http://127.0.0.1:8080/docs
python mcp_server.py                 # MCP server -> http://127.0.0.1:8000/mcp
python mcp_client.py                 # (second terminal) separate MCP client
```

`MOCK_LLM` is the default. A real model can optionally be switched on with
`USE_REAL_LLM=1 GROQ_API_KEY=...`, but every result below was produced under `MOCK_LLM`.

## Architecture

```
POST /ask ─► input_guard ─► classify_intent ──(conditional edge)──┬─► retrieve_policy (ChromaDB, 2s timeout) ─┐
             (PII mask,      (JSON memory,                        ├─► lookup_order (tool, retry policy) ──────┤
              injection)      route decision)                     └─► (blocked / clarify) ────────────────────┤
                                                                                                              ▼
                                                     finalize ◄── output_guard ◄────────────────────── generate_answer
                                              (JSON-schema check,  (groundedness                     (grounded answer /
                                               save memory)         check)                            order summary)
```

| File | Purpose |
|---|---|
| `dataset.py` | Task 1 – seeded order dataset (`ORDERS`) + validation |
| `knowledge_base/` | Task 2 – 12 policy documents |
| `rag.py` | Task 3/4 – chunking (2 strategies), embeddings, 2 ChromaDB collections, retrieval, grounded answer |
| `eval_chunking.py` | Task 5 – Precision@3 / Recall@3 for both collections |
| `calibrate.py` | Task 4 – empirical "I don't know" threshold + grounded-generation demo |
| `tools.py` | Task 6 – `check_order_status` with designed escalation score |
| `agent.py` | Task 7–10, 15, 16 – LangGraph graph, routing, memory, schema, guardrails, retry, timeouts |
| `memory.py` | Task 8 – conversation memory in `memory/sessions.json` |
| `schema.py` | Task 9 – JSON Schema validated on every response |
| `guardrails.py` | Task 10 – PII masking, injection detection, groundedness check |
| `api.py` | Task 11/12 – FastAPI (`/ask`, `/add-document`, `/health`) + JSON-Lines logging |
| `llm.py`, `eval_triad.py` | Task 13 – MOCK_LLM generator + LLM-as-judge, RAG triad on 15 queries |
| `mcp_server.py`, `mcp_client.py` | Task 14 – MCP server + separate client |
| `demo_checkpoint.py` | Task 15 – SQLite checkpoint, stop, resume in a new process |
| `demo_resilience.py` | Task 16 – retry with backoff, per-node timeout, global timeout |
| `run_all.py` | runs all of the above and saves `transcripts/` |

## Part 1 — Dataset design & RAG core

### Task 1 — Dataset design choices (reproducible)

| Choice | Value |
|---|---|
| Seed | `42` (`random.Random(seed)`) |
| Number of orders | 50 |
| Category weights | {'Apparel': 0.22, 'Beauty': 0.26, 'Footwear': 0.18, 'Home': 0.17, 'Electronics': 0.17} |
| Status weights | {'Placed': 0.15, 'Shipped': 0.25, 'Delivered': 0.35, 'Returned': 0.12, 'Refunded': 0.13} |
| `delayed_shipment` probability | 0.2 |
| `order_value_inr` ranges | {'Beauty': (199, 3999), 'Apparel': (499, 4999), 'Footwear': (799, 5999), 'Home': (299, 7999), 'Electronics': (999, 24999)} |
| `days_since_created` | uniform integer 0–30 |

**Price reasoning:** ranges follow typical Nykaa prices — beauty items run from a ₹199 lipstick to a
₹3,999 premium serum, while electronics (hair dryers, trimmers, straighteners) go up to ₹24,999.

**Validation output** (`transcripts/task01_dataset.txt`): categories {'Footwear': 14, 'Home': 11, 'Beauty': 9, 'Apparel': 8, 'Electronics': 8}; statuses {'Placed': 12, 'Delivered': 9, 'Shipped': 16, 'Returned': 6, 'Refunded': 7};
`delayed_shipment=True` = **18.0%** (required band 10–30%). No record was hand-edited.

### Task 2 — Knowledge base
12 documents in `knowledge_base/` (4 sentences each), one per required topic: return window by
category, COD refund timelines, delivery SLAs, reverse-pickup eligibility, warranty terms, order
cancellation, loyalty points, payment failure/retry, size exchange, damaged-item claims,
international shipping restrictions, and the support escalation matrix.

### Task 3 — Two chunking strategies
* **Fixed-size with overlap:** 200 characters, 50-character overlap → collection `nykaa_fixed`
* **Sentence-based:** one sentence per chunk → collection `nykaa_sentence`
* Both embedded with SentenceTransformers **`all-MiniLM-L6-v2`** (local, free), cosine space, each in its own ChromaDB collection.

### Task 4 — Calibrated "I don't know" threshold
Measured top-1 cosine similarity on the deployed collection (`nykaa_sentence`):

| In-scope query | Top-1 similarity |
|---|---|
| How many days do I have to return footwear? | 0.7029 |
| How long does a COD refund take to reach my bank account? | 0.8421 |
| How long does standard delivery take in metro cities? | 0.9264 |
| Can I cancel my order after it has been shipped? | 0.7794 |
| What is the warranty on electronics like hair dryers? | 0.8011 |
| How do I raise a claim for a damaged item? | 0.6799 |

| Out-of-scope query | Top-1 similarity |
|---|---|
| What is the capital of France? | 0.0935 |
| Can you give me a recipe for butter chicken? | 0.0909 |
| Who won the cricket world cup in 2011? | 0.1305 |

**Chosen threshold = 0.4052** — midpoint of the gap: (min in-scope 0.6799 + max out-of-scope 0.1305) / 2.
Grounded generation is demonstrated on 6 in-scope queries plus 1 out-of-scope query that triggers
the fallback in `transcripts/task04_threshold_and_grounded_generation.txt`.

### Task 5 — Chunking comparison (document-level, deduplicated)
Precision@3 = relevant unique docs / unique docs retrieved; Recall@3 = relevant unique docs / total relevant docs.

**Fixed-size with overlap (`nykaa_fixed`)**

| # | Query | Unique docs in top-3 | Arithmetic |
|---|---|---|---|
| 1 | How many days do I have to return footwear? | 01_return_window, 09_size_exchange | P = 2/2 = 1.000 · R = 2/2 = 1.000 |
| 2 | How long does a COD refund take to reach my bank account? | 02_cod_refund_timeline, 08_payment_failure_retry | P = 1/2 = 0.500 · R = 1/1 = 1.000 |
| 3 | How long does standard delivery take in metro cities? | 03_delivery_sla, 09_size_exchange | P = 1/2 = 0.500 · R = 1/1 = 1.000 |
| 4 | Can I cancel my order after it has been shipped? | 06_order_cancellation, 08_payment_failure_retry | P = 1/2 = 0.500 · R = 1/1 = 1.000 |
| 5 | What is the warranty on electronics like hair dryers? | 05_warranty_terms, 01_return_window | P = 1/2 = 0.500 · R = 1/1 = 1.000 |
| 6 | How do I raise a claim for a damaged item? | 10_damaged_item_claim | P = 1/1 = 1.000 · R = 1/1 = 1.000 |

**Mean Precision@3 = 0.667 · Mean Recall@3 = 1.0**

**Sentence-based (`nykaa_sentence`)**

| # | Query | Unique docs in top-3 | Arithmetic |
|---|---|---|---|
| 1 | How many days do I have to return footwear? | 01_return_window, 09_size_exchange | P = 2/2 = 1.000 · R = 2/2 = 1.000 |
| 2 | How long does a COD refund take to reach my bank account? | 02_cod_refund_timeline, 08_payment_failure_retry | P = 1/2 = 0.500 · R = 1/1 = 1.000 |
| 3 | How long does standard delivery take in metro cities? | 03_delivery_sla, 09_size_exchange | P = 1/2 = 0.500 · R = 1/1 = 1.000 |
| 4 | Can I cancel my order after it has been shipped? | 06_order_cancellation | P = 1/1 = 1.000 · R = 1/1 = 1.000 |
| 5 | What is the warranty on electronics like hair dryers? | 05_warranty_terms | P = 1/1 = 1.000 · R = 1/1 = 1.000 |
| 6 | How do I raise a claim for a damaged item? | 10_damaged_item_claim, 05_warranty_terms | P = 1/2 = 0.500 · R = 1/1 = 1.000 |

**Mean Precision@3 = 0.75 · Mean Recall@3 = 1.0**

**Recommendation:** I would deploy sentence-based chunking. On the same 6 queries it scored mean Precision@3 = 0.75 and mean Recall@3 = 1.0, versus 0.667 and 1.0 for fixed-size-with-overlap chunking. Sentence chunks keep each policy rule complete in one piece, so the retrieved text is cleaner to answer from.

## Part 2 — LangGraph agent

### Task 6 — `check_order_status` and escalation score
```
escalation_score = 0.6 * delayed_shipment + 0.4 * (days_since_created / 30)      # always in [0, 1]
threshold        = 0.6 + 0.4 * (median days_since_created / 30) = 0.6 + 0.4 * 13.0/30 = 0.7733
```
A late order is the main driver (weight 0.6); order age adds urgency (weight 0.4). The threshold is
the **median order age in our generated data (13.0 days)**: a delayed order is escalated once it is
older than half of all orders. Non-delayed orders can never escalate (max 0.4). In our data
6/50 orders are above the threshold: NYK-1002, NYK-1008, NYK-1010, NYK-1022, NYK-1029, NYK-1043.

### Task 7 — Graph
7 nodes (`input_guard`, `classify_intent`, `retrieve_policy`, `lookup_order`, `generate_answer`,
`output_guard`, `finalize`) and a genuine conditional edge `classify_intent → route_after_classify()`
that sends policy questions to the RAG tool and order questions to `check_order_status`.
Both routes are shown in `transcripts/task07_langgraph_routing.txt` (with the Mermaid graph text).

### Task 8 — Persisted memory
Conversation history and the last order ID are stored per `session_id` in `memory/sessions.json`.
`task08_memory_multiturn.txt`: "check order NYK-1010" → "Is it delayed?" → "Does that order need
escalation?" are answered from memory. `task08_memory_fresh_session.txt`: a new session asking
"Is it delayed?" has no remembered order, so the agent asks for the order ID.

### Task 9 — Structured output
`schema.py` defines a JSON Schema (route enum, answer, sources, order, guardrails, nodes executed).
The `finalize` node validates **every** response with `jsonschema.validate`; the API also enforces it
with Pydantic models. See `transcripts/task09_structured_output.txt`.

### Task 10 — Guardrails (each shown firing in `transcripts/task10_guardrails.txt`)
* **PII masking (input):** phone numbers and card last-4 digits → `[PHONE_MASKED]`, `[CARD_LAST4_MASKED]`
  before the text reaches the model, memory or logs. Customer name / address are free text and out of scope.
* **Prompt-injection detection (input):** e.g. "Ignore all previous instructions…" → route `blocked`.
* **Groundedness check (output):** the agent refuses unless (1) top-1 similarity clears the calibrated
  threshold, (2) the retrieved context covers at least 50% of the question's key words, and (3) every
  answer sentence has at least 80% of its content words in the retrieved context.
  Demonstrated with a simulated hallucinating model and with an unsupported question ("Does Nykaa offer car insurance?").

## Part 3 — Evaluation, observability & deployment

### Task 11 — FastAPI
`POST /ask`, `POST /add-document`, `GET /health` with Pydantic request/response models
(`uvicorn api:app --port 8080`, Swagger at `/docs`). Exercised in `transcripts/task11_12_fastapi_and_logging.txt`.

### Task 12 — Structured logging
Each request appends one JSON line to `logs/requests.jsonl` with `trace_id`, timestamp, endpoint,
latency and route. The request text is masked with the same `mask_pii()` used by the agent, and the
demo asserts that the raw phone number / card digits never appear in the log file.

### Task 13 — RAG triad (LLM-as-judge under MOCK_LLM)
Judge prompt: `llm.JUDGE_PROMPT`. Under `MOCK_LLM` it is scored deterministically: context relevance =
mean cosine(query, retrieved chunks); groundedness = share of answer sentences supported by the
context (a refusal makes no claims → 1.0); answer relevance = cosine(query, answer).

| # | Topic | Query | Context rel. | Groundedness | Answer rel. |
|---|---|---|---|---|---|
| 1 | return window by category | Within how many days can I return a sealed lipstick or Electronics item? | 0.543 | 1.0 | 0.68 |
| 2 | COD refund timelines | I paid cash on delivery, how will I get my refund and how long does it take? | 0.624 | 1.0 | 0.64 |
| 3 | delivery SLAs | How many business days does delivery take to a non-metro pin code? | 0.65 | 1.0 | 0.716 |
| 4 | reverse-pickup eligibility | What happens if reverse pickup is not available at my pin code? | 0.597 | 1.0 | 0.694 |
| 5 | warranty terms by category | Do apparel or beauty products come with a warranty? | 0.574 | 1.0 | 0.649 |
| 6 | order-cancellation policy | Can I cancel an order while its status is Placed? | 0.697 | 1.0 | 0.662 |
| 7 | loyalty-points redemption | How much is one loyalty point worth and when do points expire? | 0.683 | 1.0 | 0.731 |
| 8 | payment failure / retry | My payment failed but money was debited, when will I get it back? | 0.582 | 1.0 | 0.583 |
| 9 | size-exchange policy | Can I exchange my shoes for a different size? | 0.539 | 1.0 | 0.602 |
| 10 | damaged-item claim process | My item arrived broken, how many hours do I have to raise a claim? | 0.568 | 1.0 | 0.644 |
| 11 | international shipping restrictions | Can I ship perfume internationally to the UK? | 0.53 | 1.0 | 0.694 |
| 12 | support escalation matrix | When is my complaint escalated to the grievance officer? | 0.566 | 1.0 | 0.541 |
| 13 | OUT-OF-SCOPE | What is the best laptop for gaming in 2026? | 0.166 | 1.0 | 0.059 |
| 14 | OUT-OF-SCOPE | Tell me a joke about cats. | 0.103 | 1.0 | 0.077 |
| 15 | EDGE CASE (vague) | help | 0.18 | 1.0 | 0.039 |

**Averages over 15 queries: context relevance = 0.507 · groundedness = 1.0 · answer relevance = 0.534**

## Part 4 — Resilience & interoperability

### Task 14 — MCP
`mcp_server.py` wraps `check_order_status` (with docstring) as a FastMCP tool served at
`http://127.0.0.1:8000/mcp`. `mcp_client.py` is a separate process that calls it for `NYK-1010` and
`NYK-1004` and prints the standardized MCP responses (`transcripts/task14_mcp.txt`).

### Task 15 — SQLite checkpointing
Graph compiled with `AsyncSqliteSaver` (`checkpoints.sqlite`, from `langgraph-checkpoint-sqlite`) keyed by
`thread_id`. Process A runs `input_guard → classify_intent → retrieve_policy` and is stopped before
`generate_answer`; a **new process B** resumes the same thread, loads the saved state, runs only the
remaining 3 nodes and asserts none of the first 3 were re-executed (`transcripts/task15_checkpointing.txt`).

### Task 16 — Timeouts & retries
* **Retry policy** on `lookup_order`: max_attempts = 4, initial_interval = 0.2 s, backoff_factor = 2,
  max_interval = 2 s, jitter = on. A simulated outage fails the first 2 calls; the 3rd succeeds.
* **Per-node timeout** on `retrieve_policy`: 2 s (LangGraph `TimeoutPolicy`) – a 5 s simulated call fails cleanly with `NodeTimeoutError`.
* **Global timeout** on the whole graph: 10 s by default (`asyncio.wait_for`); the demo uses a 3 s budget with
  1 s nodes and the run is cancelled at ~3 s with `GraphTimeoutError`.

See `transcripts/task16_retries_and_timeouts.txt`.

## Transcripts
All outputs are in `transcripts/` (regenerate with `python run_all.py`).
