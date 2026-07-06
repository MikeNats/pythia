"""Offline RAG evals — LLM-judge scoring against a golden set.

Needs the DB **and** Ollama running (`ollama serve` + `ollama pull llama3.2`).

    ./run eval

Two independent LLM judges (reference-free, RAGAS-style), each run N times and
decided by majority vote (self-consistency — smooths a small local judge):

  - context relevance : do the RETRIEVED chunks contain what's needed to answer
                        the question?  → grades RETRIEVAL
  - faithfulness      : is the ANSWER supported by those chunks (no made-up
                        facts)?        → grades GENERATION

PASS = both judges pass. Ceiling note: a small local judge caps reliability;
production would swap in a stronger judge model (costs money).
"""

import asyncio
import sys

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import SessionLocal
from app.ingest.chunking import chunk_text
from app.ingest.models import Chunk, Document, SourceType
from app.llm.router import Client, LLMProviders, resolve_client
from app.retrieval.embedder import get_embedder
from app.retrieval.services import search_query_to_chunks, single_query_to_answer

VOTES = 3  # judge runs per check; majority wins (self-consistency)

# Both models are passed dynamically (CLI args), defaulting to llama3.2. A weak
# 3B model caps quality for BOTH tasks — a stronger LOCAL model is far better
# (still $0):
#   ./run eval <judge_model> <answer_model>
#   ./run eval qwen2.5:7b qwen2.5:7b
DEFAULT_JUDGE_MODEL = "llama3.2"
DEFAULT_ANSWER_MODEL = "llama3.2"

# A mixed corpus: 1 relevant doc + 2 distractors. Ingested together every run,
# so retrieval must pick the right chunk out of noise — a realistic test.
RELEVANT_DOC = """
Acme Corp return policy. Customers may return any item within 30 days of
purchase for a full refund. Shipping is free on all orders over 50 dollars.
Standard delivery takes 3 to 5 business days. Our support team is available
Monday to Friday, from 9am to 5pm Eastern time.
"""
PREGNANCY_DOC = """
Stages of pregnancy. Pregnancy lasts about 40 weeks, grouped into three
trimesters. In the third trimester the baby gains weight and may turn head-down
for birth. Most full-term babies weigh between 6 and 9 pounds at delivery.
"""
SOURDOUGH_DOC = """
Basic sourdough bread. Mix flour, water, salt, and an active starter. Let the
dough ferment 4 to 12 hours, shape it, and proof overnight in the fridge. Bake
at 230C in a covered dutch oven for 45 minutes until the crust is deep brown.
"""

CORPUS: list[tuple[str, str]] = [
    ("acme-return-policy", RELEVANT_DOC),
    ("pregnancy-stages", PREGNANCY_DOC),
    ("sourdough-recipe", SOURDOUGH_DOC),
]

GOLDEN: list[str] = [
    "How many days do I have to return an item?",
    "When do I get free shipping?",
    "How long does standard delivery take?",
    # tricky: 3 weeks = 21 days ≤ 30 → yes.
    "I bought an item 3 weeks ago — can I still return it for a full refund?",
]

CONTEXT_RELEVANCE = (
    "You grade RETRIEVAL. Given a QUESTION and the retrieved CHUNKS, decide if "
    "the chunks contain the information needed to answer the question. Use only "
    "the chunks — no outside knowledge."
)
FAITHFULNESS = (
    "You grade an ANSWER. Given the CONTEXT and an ANSWER, decide if the answer "
    "is fully supported by the context (no invented facts). Use only the context."
)


ADVERSARY = (
    "You are a skeptical adversary. Try HARD to REFUTE the CLAIM using ONLY the "
    "EVIDENCE — hunt for any unsupported part, gap, or contradiction. Set "
    "refuted=true only if you find a real flaw; if the claim genuinely holds "
    "against the evidence, set refuted=false."
)


class Verdict(BaseModel):
    passed: bool
    reason: str


class Attack(BaseModel):
    refuted: bool
    reason: str


async def judge(client: Client, *, criteria: str, content: str) -> Verdict:
    """Layer 1 — neutral judge, run VOTES times; majority verdict + one reason."""
    verdicts: list[Verdict] = []
    for _ in range(VOTES):
        resp = await client.inference(
            name="judge",
            system_prompt=criteria,
            user_prompt=content,
            tool_schema=Verdict.model_json_schema(),
        )
        verdicts.append(Verdict.model_validate(resp.content))
    passed = sum(v.passed for v in verdicts) > VOTES // 2
    reason = next((v.reason for v in verdicts if v.passed == passed), "")
    return Verdict(passed=passed, reason=reason)


async def adversary(client: Client, *, claim: str, evidence: str) -> Attack:
    attacks: list[Attack] = []
    for _ in range(VOTES):
        resp = await client.inference(
            name="adversary",
            system_prompt=ADVERSARY,
            user_prompt=f"CLAIM:\n{claim}\n\nEVIDENCE:\n{evidence}",
            tool_schema=Attack.model_json_schema(),
        )
        attacks.append(Attack.model_validate(resp.content))
    refuted = sum(a.refuted for a in attacks) > VOTES // 2
    reason = next((a.reason for a in attacks if a.refuted == refuted), "")
    return Attack(refuted=refuted, reason=reason)


async def _ingest_corpus(session: AsyncSession) -> None:
    """Chunk + embed every doc in CORPUS and flush (visible to queries, NOT committed).

    Ingests the relevant doc alongside distractors, so retrieval is tested
    against a mixed corpus. Flushed only → rolled back on close (no pollution).
    """
    embedder = get_embedder()
    for name, text in CORPUS:
        pieces = chunk_text(text)
        vectors = await asyncio.to_thread(embedder.embed, pieces)
        session.add(
            Document(
                name=name,
                source_type=SourceType.upload,
                source_ref="eval",
                content_type="text/plain",
                byte_size=len(text.encode("utf-8")),
                chunks=[
                    Chunk(position=i, text=p, embedding=v)
                    for i, (p, v) in enumerate(zip(pieces, vectors, strict=True))
                ],
            )
        )
    await session.flush()


async def run(judge_model: str, answer_model: str) -> None:
    passed = 0
    async with SessionLocal() as session:
        await _ingest_corpus(session)
        answer_client = resolve_client(LLMProviders.OLLAMA, answer_model)
        judge_client = resolve_client(LLMProviders.OLLAMA, judge_model)

        print(
            f"\n=== RAG eval (answer={answer_model}, judge={judge_model}, "
            f"majority of {VOTES}) ===\n"
        )
        for question in GOLDEN:
            hits = await search_query_to_chunks(session, question, k=3)
            chunks = "\n".join(f"[{i}] {h.text}" for i, h in enumerate(hits, start=1))
            result = await single_query_to_answer(session, question, answer_client, k=3)
            answer = result.answer

            # Layer 1 — neutral judges + voting (unchanged)
            retrieval = await judge(
                judge_client,
                criteria=CONTEXT_RELEVANCE,
                content=f"QUESTION:\n{question}\n\nCHUNKS:\n{chunks}",
            )
            grounding = await judge(
                judge_client,
                criteria=FAITHFULNESS,
                content=f"CONTEXT:\n{chunks}\n\nANSWER:\n{answer}",
            )
            layer1 = retrieval.passed and grounding.passed

            # Layer 2 — adversarial: only challenge what survived layer 1
            attack = Attack(refuted=False, reason="skipped (failed layer 1)")
            if layer1:
                attack = await adversary(
                    judge_client,
                    claim=f"The answer '{answer}' is fully supported by the chunks.",
                    evidence=chunks,
                )

            ok = layer1 and not attack.refuted
            passed += int(ok)

            print(f"{'✅' if ok else '❌'} {question}")
            print(f"     answer            : {answer[:80]!r}")
            print("     retrieved (score → full chunk):")
            for h in hits:
                # whitespace-collapsed so the FULL chunk shows on one line
                print(f"       [{h.score:.2f}] {' '.join(h.text.split())}")
            print(
                f"  L1 context_relevance : {retrieval.passed} — {retrieval.reason[:60]}"
            )
            print(
                f"  L1 faithfulness      : {grounding.passed} — {grounding.reason[:60]}"
            )
            print(f"  L2 adversary refuted : {attack.refuted} — {attack.reason[:60]}")
            print()

    print(f"score: {passed}/{len(GOLDEN)} passed\n")


if __name__ == "__main__":
    # ./run eval [judge_model] [answer_model]   — each defaults to llama3.2
    judge_model = (
        sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else DEFAULT_JUDGE_MODEL
    )
    answer_model = (
        sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else DEFAULT_ANSWER_MODEL
    )
    asyncio.run(run(judge_model, answer_model))
