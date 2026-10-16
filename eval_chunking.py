"""
eval_chunking.py - Task 5: compare fixed-size vs sentence-based chunking.

For each query: take the top-3 chunks, map every chunk to its parent document,
de-duplicate, then score at the DOCUMENT level:
    Precision@3 = relevant docs retrieved / unique docs retrieved
    Recall@3    = relevant docs retrieved / total relevant docs
"""
import json

import config
from eval_queries import IN_SCOPE
from rag import ensure_index, retrieve


def evaluate(collection: str) -> dict:
    rows = []
    for item in IN_SCOPE:
        chunks = retrieve(item["query"], collection, k=3)
        docs = list(dict.fromkeys(c["doc_id"] for c in chunks))     # dedup, keep order
        relevant = set(item["relevant"])
        hits = len(relevant & set(docs))
        p, r = hits / len(docs), hits / len(relevant)
        rows.append({"query": item["query"], "top3_chunk_docs": [c["doc_id"] for c in chunks],
                     "unique_docs": docs, "relevant": sorted(relevant), "hits": hits,
                     "precision@3": round(p, 3), "recall@3": round(r, 3),
                     "arithmetic": f"P = {hits}/{len(docs)} = {p:.3f} | R = {hits}/{len(relevant)} = {r:.3f}"})
    mean_p = sum(x["precision@3"] for x in rows) / len(rows)
    mean_r = sum(x["recall@3"] for x in rows) / len(rows)
    return {"collection": collection, "rows": rows,
            "mean_precision@3": round(mean_p, 3), "mean_recall@3": round(mean_r, 3)}


def main():
    ensure_index()
    results = {name: evaluate(name) for name in (config.COLLECTION_FIXED, config.COLLECTION_SENTENCE)}
    for res in results.values():
        print(f"\n=== Collection: {res['collection']} ===")
        for i, row in enumerate(res["rows"], 1):
            print(f"Q{i}: {row['query']}")
            print(f"    top-3 chunks from docs : {row['top3_chunk_docs']}")
            print(f"    unique docs (deduped)  : {row['unique_docs']}")
            print(f"    relevant docs          : {row['relevant']}")
            print(f"    {row['arithmetic']}")
        print(f"MEAN Precision@3 = {res['mean_precision@3']} | MEAN Recall@3 = {res['mean_recall@3']}")

    fx, se = results[config.COLLECTION_FIXED], results[config.COLLECTION_SENTENCE]
    score = lambda r: r["mean_precision@3"] + r["mean_recall@3"]
    winner, loser = (se, fx) if score(se) >= score(fx) else (fx, se)
    nice = {config.COLLECTION_FIXED: "fixed-size-with-overlap", config.COLLECTION_SENTENCE: "sentence-based"}
    recommendation = (
        f"I would deploy {nice[winner['collection']]} chunking. On the same {len(IN_SCOPE)} queries it scored "
        f"mean Precision@3 = {winner['mean_precision@3']} and mean Recall@3 = {winner['mean_recall@3']}, versus "
        f"{loser['mean_precision@3']} and {loser['mean_recall@3']} for {nice[loser['collection']]} chunking. "
        + ("Sentence chunks keep each policy rule complete in one piece, so the retrieved text is cleaner to answer from."
           if winner is se else
           "Fixed-size chunks with overlap keep neighbouring sentences together, which gave more context per hit.")
    )
    print("\nRECOMMENDATION:\n" + recommendation)
    config.CHUNKING_EVAL_FILE.parent.mkdir(exist_ok=True)
    config.CHUNKING_EVAL_FILE.write_text(json.dumps(
        {"recommended_collection": winner["collection"], "recommendation": recommendation,
         "results": results}, indent=2))


if __name__ == "__main__":
    main()
