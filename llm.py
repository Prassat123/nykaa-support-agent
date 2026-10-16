"""
llm.py - the language-model layer.

DEFAULT = MOCK_LLM (deterministic, zero API keys, zero network).
  * generate_answer(): extractive - picks the retrieved sentences that best match the
    question, so every word of the answer comes from the retrieved context.
  * judge(): LLM-as-judge for the RAG triad, scored deterministically
    (embedding similarity + word overlap) so results are reproducible.

OPTIONAL real LLM: export USE_REAL_LLM=1 and GROQ_API_KEY=... (Groq, OpenAI-compatible).
"""
import json
import os

import config
from rag import cosine, embed, split_sentences, tokenize

GROUNDED_PROMPT = """You are Nykaa's customer-support assistant.
Answer the question using ONLY the context below. If the context does not contain
the answer, reply exactly: I don't know.

Context:
{context}

Question: {question}
Answer:"""

JUDGE_PROMPT = """You are an impartial evaluator of a RAG system. Score each metric from 0 to 1.
- context_relevance: how relevant is the retrieved CONTEXT to the QUESTION?
- groundedness: is every claim in the ANSWER supported by the CONTEXT?
- answer_relevance: how well does the ANSWER address the QUESTION?
Return JSON: {{"context_relevance": x, "groundedness": y, "answer_relevance": z}}

QUESTION: {question}
CONTEXT: {context}
ANSWER: {answer}"""


# ---------------------------------------------------------------- optional real LLM
def _call_groq(prompt: str) -> str:
    import httpx
    r = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        json={"model": config.GROQ_MODEL, "temperature": 0,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------- generation
def _mock_generate(question: str, chunks: list[dict], max_sentences: int = 2) -> str:
    """Extractive answer: best-matching retrieved sentence (+ one more from the SAME document)."""
    q_words = set(tokenize(question))
    candidates = []
    for rank, chunk in enumerate(chunks):
        for pos, sent in enumerate(split_sentences(chunk["text"])):
            overlap = len(q_words & set(tokenize(sent)))
            # more shared words first, then higher-ranked chunk, then earlier sentence
            candidates.append((-overlap, rank, pos, sent, chunk["doc_id"]))
    candidates.sort()
    best = candidates[0]
    picked = [best[3]]
    for neg_overlap, _, _, sent, doc_id in candidates[1:]:
        if len(picked) == max_sentences:
            break
        if doc_id == best[4] and neg_overlap < 0 and sent not in picked:
            picked.append(sent)
    return "According to Nykaa policy: " + " ".join(picked)


def generate_answer(question: str, chunks: list[dict], hallucinate: bool = False) -> str:
    """Generate an answer from retrieved chunks only.

    `hallucinate=True` is ONLY for the guardrail demo (Task 10): it simulates a model
    that adds an unsupported claim, so we can show the output-side check catching it.
    """
    if config.MOCK_LLM:
        answer = _mock_generate(question, chunks)
    else:
        context = "\n".join(f"- {c['text']}" for c in chunks)
        answer = _call_groq(GROUNDED_PROMPT.format(context=context, question=question))
    if hallucinate:
        answer += " Nykaa also gives every customer a free gold membership worth INR 5000."
    return answer


# ---------------------------------------------------------------- Task 13 judge
def _support_ratio(answer: str, context: str) -> float:
    """Share of answer sentences whose content words mostly appear in the context."""
    ctx = set(tokenize(context))
    body = answer.replace("According to Nykaa policy:", "").strip()
    sents = [s for s in split_sentences(body) if tokenize(s)]
    if not sents:
        return 1.0
    supported = sum(1 for s in sents
                    if len(set(tokenize(s)) & ctx) / len(set(tokenize(s))) >= 0.8)
    return supported / len(sents)


def judge(question: str, context_chunks: list[dict], answer: str, is_fallback: bool) -> dict:
    """Score context relevance, groundedness and answer relevance (each 0..1)."""
    context = " ".join(c["text"] for c in context_chunks)
    if not config.MOCK_LLM:
        raw = _call_groq(JUDGE_PROMPT.format(question=question, context=context, answer=answer))
        return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])

    q_vec = embed([question])[0]
    if context_chunks:
        ctx_vecs = embed([c["text"] for c in context_chunks])
        context_relevance = max(0.0, sum(cosine(q_vec, v) for v in ctx_vecs) / len(ctx_vecs))
    else:
        context_relevance = 0.0
    # a refusal makes no unsupported claim, so it is fully grounded
    groundedness = 1.0 if is_fallback else _support_ratio(answer, context)
    answer_relevance = max(0.0, cosine(q_vec, embed([answer])[0]))
    return {"context_relevance": round(context_relevance, 3),
            "groundedness": round(groundedness, 3),
            "answer_relevance": round(answer_relevance, 3)}
