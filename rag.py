"""
rag.py - Tasks 3 & 4: chunking, embedding, ChromaDB indexing, retrieval and
grounded generation.

Flow:  documents -> chunks (2 strategies) -> embeddings -> 2 Chroma collections
       query -> embedding -> top-k chunks -> similarity check -> answer from chunks
"""
import re
from functools import lru_cache

import chromadb

import config

STOPWORDS = set("""a an the is are was were be been of to in on for and or with by at from as
it its this that these those my me i we you your our can do does did how what when where which
who why will would should could may might much many any some there their they them his her
than then so if not no yes about into over after before within per get got have has had""".split())


# ---------------------------------------------------------------- text helpers
SYNONYMS = {"shoes": "footwear", "shoe": "footwear", "sneakers": "footwear", "clothes": "apparel",
            "dress": "apparel", "shirt": "apparel", "makeup": "beauty", "lipstick": "beauty",
            "cod": "cash", "abroad": "international", "broken": "damaged"}


def _norm(word: str) -> str:
    word = SYNONYMS.get(word, word)
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def tokenize(text: str) -> list[str]:
    """Lower-case content words, lightly normalised (plural -> singular, a few synonyms).
    Used by the mock LLM and the groundedness checks - NOT by retrieval (that uses embeddings)."""
    return [_norm(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? followed by whitespace."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


# ---------------------------------------------------------------- documents
def load_documents() -> dict[str, str]:
    """doc_id (file name without .txt) -> full text."""
    return {p.stem: p.read_text(encoding="utf-8").strip()
            for p in sorted(config.KB_DIR.glob("*.txt"))}


# ---------------------------------------------------------------- Task 3: chunking
def chunk_fixed(text: str, size: int = config.FIXED_CHUNK_SIZE,
                overlap: int = config.FIXED_CHUNK_OVERLAP) -> list[str]:
    """Fixed-size chunks of `size` characters; consecutive chunks share `overlap` characters."""
    chunks, start, step = [], 0, size - overlap
    while start < len(text):
        chunks.append(text[start:start + size].strip())
        if start + size >= len(text):
            break
        start += step
    return chunks


def chunk_sentences(text: str) -> list[str]:
    """Sentence-based chunking: one sentence = one chunk."""
    return split_sentences(text)


STRATEGIES = {
    config.COLLECTION_FIXED: chunk_fixed,
    config.COLLECTION_SENTENCE: chunk_sentences,
}


# ---------------------------------------------------------------- embeddings
@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBED_MODEL, device="cpu")  # CPU = same results on every machine


def embed(texts: list[str]) -> list[list[float]]:
    vectors = get_embedder().encode(texts, normalize_embeddings=True)
    return [list(map(float, v)) for v in vectors]


def cosine(a: list[float], b: list[float]) -> float:
    """Embeddings are normalised, so cosine similarity = dot product."""
    return float(sum(x * y for x, y in zip(a, b)))


# ---------------------------------------------------------------- ChromaDB
@lru_cache(maxsize=1)
def get_client():
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(name: str):
    return get_client().get_or_create_collection(
        name, metadata={"hnsw:space": "cosine"}, embedding_function=None)


def _add_chunks(collection_name: str, doc_id: str, text: str) -> int:
    chunks = STRATEGIES[collection_name](text)
    get_collection(collection_name).upsert(
        ids=[f"{doc_id}::{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks),
        metadatas=[{"doc_id": doc_id, "chunk_index": i, "strategy": collection_name}
                   for i in range(len(chunks))],
    )
    return len(chunks)


def build_index(reset: bool = True) -> dict[str, int]:
    """Chunk + embed every document with BOTH strategies, each in its own collection."""
    client, docs, counts = get_client(), load_documents(), {}
    for name in STRATEGIES:
        if reset and name in [c.name if hasattr(c, "name") else c for c in client.list_collections()]:
            client.delete_collection(name)
        counts[name] = sum(_add_chunks(name, d, t) for d, t in docs.items())
    return counts


def ensure_index():
    """Build the index only if it does not exist yet. Also loads the embedding model once
    up front, so the slow first-time model load never counts against a node's timeout."""
    embed(["warm-up"])
    try:
        if get_collection(config.COLLECTION_SENTENCE).count() and get_collection(config.COLLECTION_FIXED).count():
            return
    except Exception:
        pass
    build_index()


def add_document(doc_id: str, text: str) -> dict[str, int]:
    """Save a new policy document and index it in both collections (used by POST /add-document)."""
    doc_id = re.sub(r"[^a-zA-Z0-9_-]", "_", doc_id)
    (config.KB_DIR / f"{doc_id}.txt").write_text(text.strip() + "\n", encoding="utf-8")
    return {name: _add_chunks(name, doc_id, text.strip()) for name in STRATEGIES}


def remove_document(doc_id: str) -> None:
    """Delete a document file and its chunks from both collections."""
    (config.KB_DIR / f"{doc_id}.txt").unlink(missing_ok=True)
    for name in STRATEGIES:
        get_collection(name).delete(where={"doc_id": doc_id})


def retrieve(query: str, collection_name: str = config.DEPLOYED_COLLECTION,
             k: int = config.TOP_K) -> list[dict]:
    """Top-k chunks with cosine similarity (1 - cosine distance)."""
    res = get_collection(collection_name).query(query_embeddings=embed([query]), n_results=k)
    return [{"text": doc, "doc_id": meta["doc_id"], "similarity": round(1 - dist, 4)}
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]


# ---------------------------------------------------------------- Task 4: grounded generation
FALLBACK_ANSWER = ("I don't know. I could not find this in Nykaa's policy documents, "
                   "so I will not guess. Please contact Nykaa support for help.")


def grounded_answer(query: str, collection_name: str = config.DEPLOYED_COLLECTION,
                    threshold: float | None = None, chunks: list[dict] | None = None) -> dict:
    """Answer using ONLY retrieved chunks; fall back to 'I don't know' below the threshold."""
    from llm import generate_answer
    threshold = config.load_threshold() if threshold is None else threshold
    chunks = retrieve(query, collection_name) if chunks is None else chunks
    top1 = chunks[0]["similarity"] if chunks else 0.0
    if top1 < threshold:
        return {"answer": FALLBACK_ANSWER, "sources": [], "top1_similarity": top1,
                "threshold": threshold, "fallback": True, "chunks": chunks}
    supporting = [c for c in chunks if c["similarity"] >= threshold]
    answer = generate_answer(query, supporting)
    return {"answer": answer, "sources": sorted({c["doc_id"] for c in supporting}),
            "top1_similarity": top1, "threshold": threshold, "fallback": False, "chunks": supporting}


if __name__ == "__main__":
    # Task 2 + Task 3 transcript
    docs = load_documents()
    print(f"Task 2 - knowledge base: {len(docs)} documents in knowledge_base/")
    for d, t in docs.items():
        print(f"  {d:34} {len(split_sentences(t))} sentences, {len(t)} chars")
    print(f"\nTask 3 - fixed-size chunking: size={config.FIXED_CHUNK_SIZE} chars, "
          f"overlap={config.FIXED_CHUNK_OVERLAP} chars | sentence chunking: 1 sentence per chunk")
    sample = next(iter(docs))
    print(f"\nExample - '{sample}' fixed-size chunks:")
    for c in chunk_fixed(docs[sample]):
        print(f"   [{c}]")
    print(f"Example - '{sample}' sentence chunks:")
    for c in chunk_sentences(docs[sample]):
        print(f"   [{c}]")
    counts = build_index()
    print(f"\nEmbedded with SentenceTransformers '{config.EMBED_MODEL}' and indexed in 2 separate "
          f"ChromaDB collections: {counts}")
    q = "Can I return a pair of shoes after two weeks?"
    for name in STRATEGIES:
        print(f"\nSample query on '{name}': {q}")
        for r in retrieve(q, name):
            print(f"   sim={r['similarity']:.4f}  {r['doc_id']:30} | {r['text'][:90]}")
