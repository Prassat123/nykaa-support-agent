"""
config.py - one place for every setting used by the project.

MOCK_LLM is ON by default: the project needs zero API keys and zero network
access for the language-model parts (answer generation + LLM judge).
Set USE_REAL_LLM=1 and GROQ_API_KEY=... to optionally use a real model.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------- folders / files ----------
KB_DIR = ROOT / "knowledge_base"          # Task 2 documents
CHROMA_DIR = ROOT / "chroma_db"           # Task 3 vector index (generated)
MEMORY_FILE = ROOT / "memory" / "sessions.json"   # Task 8 conversation memory
LOG_FILE = ROOT / "logs" / "requests.jsonl"       # Task 12 structured logs
CHECKPOINT_DB = ROOT / "checkpoints.sqlite"       # Task 15 LangGraph checkpoints
THRESHOLD_FILE = ROOT / "results" / "threshold.json"  # Task 4 calibrated value
TRANSCRIPTS_DIR = ROOT / "transcripts"            # proof that every task runs

# ---------- models ----------
EMBED_MODEL = "all-MiniLM-L6-v2"          # free, local SentenceTransformers model
MOCK_LLM = os.getenv("USE_REAL_LLM", "0") != "1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ---------- chunking (Task 3) ----------
FIXED_CHUNK_SIZE = 200      # characters per fixed-size chunk
FIXED_CHUNK_OVERLAP = 50    # characters shared between neighbouring chunks
COLLECTION_FIXED = "nykaa_fixed"
COLLECTION_SENTENCE = "nykaa_sentence"
TOP_K = 3

# Which collection the agent uses: the winner of the Task 5 comparison
# (results/chunking_eval.json). Sentence-based is used until that file exists.
CHUNKING_EVAL_FILE = ROOT / "results" / "chunking_eval.json"
DEPLOYED_COLLECTION = (json.loads(CHUNKING_EVAL_FILE.read_text())["recommended_collection"]
                       if CHUNKING_EVAL_FILE.exists() else COLLECTION_SENTENCE)

# ---------- resilience (Task 16) ----------
NODE_TIMEOUT_SECONDS = 2.0      # per-node timeout on the retrieve_policy node
GLOBAL_TIMEOUT_SECONDS = 10.0   # whole-graph timeout for one request
RETRY_MAX_ATTEMPTS = 4
RETRY_INITIAL_INTERVAL = 0.2    # seconds
RETRY_BACKOFF_FACTOR = 2.0
RETRY_MAX_INTERVAL = 2.0        # seconds
RETRY_JITTER = True


def load_threshold(default: float = 0.35) -> float:
    """Read the empirically calibrated 'I don't know' threshold (Task 4)."""
    if THRESHOLD_FILE.exists():
        return float(json.loads(THRESHOLD_FILE.read_text())["threshold"])
    return default
