"""
eval_triad.py - Task 13: RAG triad evaluation with an LLM-as-judge (MOCK_LLM).

15 queries: one per required KB topic (12) + 3 out-of-scope / edge cases.
For each query we score:
  context_relevance - is the retrieved context relevant to the question?
  groundedness      - is the answer supported by the retrieved context?
  answer_relevance  - does the answer address the question?
The judge prompt is llm.JUDGE_PROMPT. Under MOCK_LLM it is scored deterministically
(embedding cosine similarity + word-overlap support ratio), so results are reproducible.
"""
import json

import config
from eval_queries import TRIAD_QUERIES
from llm import judge
from rag import ensure_index, grounded_answer, retrieve


def main():
    ensure_index()
    print("Judge prompt template used:\n" + __import__("llm").JUDGE_PROMPT + "\n")
    rows = []
    for i, (topic, q) in enumerate(TRIAD_QUERIES, 1):
        retrieved = retrieve(q, config.DEPLOYED_COLLECTION)
        result = grounded_answer(q, chunks=retrieved)
        scores = judge(q, retrieved, result["answer"], result["fallback"])
        rows.append({"id": i, "topic": topic, "query": q, "answer": result["answer"],
                     "fallback": result["fallback"], **scores})

    header = f"{'#':>2} | {'topic':36} | {'ctx_rel':>7} | {'ground':>6} | {'ans_rel':>7} | fallback"
    print(header + "\n" + "-" * len(header))
    for r in rows:
        print(f"{r['id']:>2} | {r['topic'][:36]:36} | {r['context_relevance']:7.3f} | "
              f"{r['groundedness']:6.3f} | {r['answer_relevance']:7.3f} | {r['fallback']}")
    avg = {k: round(sum(r[k] for r in rows) / len(rows), 3)
           for k in ("context_relevance", "groundedness", "answer_relevance")}
    print("-" * len(header))
    print(f"AVERAGE (15 queries): context_relevance = {avg['context_relevance']} | "
          f"groundedness = {avg['groundedness']} | answer_relevance = {avg['answer_relevance']}")

    print("\nPer-query answers:")
    for r in rows:
        print(f"{r['id']:>2}. Q: {r['query']}\n    A: {r['answer']}")

    out = config.ROOT / "results" / "rag_triad.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "averages": avg}, indent=2))


if __name__ == "__main__":
    main()
