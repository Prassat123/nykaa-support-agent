"""
calibrate.py - Task 4: empirically calibrate the "I don't know" threshold, then
demonstrate grounded generation on in-scope queries + an out-of-scope query.

Under MOCK_LLM, retrieval similarity is the only groundedness signal, so we MEASURE
top-1 cosine similarity for in-scope and out-of-scope queries and put the threshold
in the gap between the two clusters (no untested 0.5/0.6/0.7 preset).
"""
import json

import config
from eval_queries import IN_SCOPE, OUT_OF_SCOPE
from rag import ensure_index, grounded_answer, retrieve


def main():
    ensure_index()
    coll = config.DEPLOYED_COLLECTION
    print(f"Collection used (winner of Task 5): {coll}\n")

    print("Step 1 - measure top-1 cosine similarity")
    in_scores, out_scores = [], []
    for item in IN_SCOPE:
        s = retrieve(item["query"], coll, k=1)[0]["similarity"]
        in_scores.append(s)
        print(f"  IN-SCOPE   {s:.4f}  {item['query']}")
    for q in OUT_OF_SCOPE:
        s = retrieve(q, coll, k=1)[0]["similarity"]
        out_scores.append(s)
        print(f"  OUT-SCOPE  {s:.4f}  {q}")

    lo_in, hi_out = min(in_scores), max(out_scores)
    if lo_in > hi_out:
        threshold = round((lo_in + hi_out) / 2, 4)
        rule = f"midpoint of the gap: (min in-scope {lo_in:.4f} + max out-of-scope {hi_out:.4f}) / 2"
    else:  # clusters overlap - fall back to the midpoint of the two means
        m_in, m_out = sum(in_scores) / len(in_scores), sum(out_scores) / len(out_scores)
        threshold = round((m_in + m_out) / 2, 4)
        rule = f"clusters overlap, so midpoint of means ({m_in:.4f} + {m_out:.4f}) / 2"
    print(f"\nStep 2 - chosen threshold = {threshold}  ({rule})")

    config.THRESHOLD_FILE.parent.mkdir(exist_ok=True)
    config.THRESHOLD_FILE.write_text(json.dumps({
        "collection": coll, "threshold": threshold, "rule": rule,
        "in_scope_top1": dict(zip([i["query"] for i in IN_SCOPE], in_scores)),
        "out_of_scope_top1": dict(zip(OUT_OF_SCOPE, out_scores))}, indent=2))

    print("\nStep 3 - grounded generation demo (answers use ONLY retrieved chunks)")
    for q in [i["query"] for i in IN_SCOPE] + [OUT_OF_SCOPE[0]]:
        r = grounded_answer(q, coll, threshold)
        print(f"\nQ: {q}\n   top-1 similarity = {r['top1_similarity']} | threshold = {threshold} | "
              f"fallback = {r['fallback']} | sources = {r['sources']}\n   A: {r['answer']}")


if __name__ == "__main__":
    main()
