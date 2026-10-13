"""Optional style-blind LLM judge (disabled by default).

The judge never sees the target style; it rates only coherence/fluency to avoid the
documented style-bias circularity (design doc §6.1). Requires `openai` and an API key.
"""
from __future__ import annotations

from typing import Optional, Sequence

RUBRIC = (
    "You are evaluating text quality only. Do not reward any particular writing style, "
    "era, or author. Rate the coherence and fluency of the passage on a 1-5 scale "
    "(1 = incoherent/repetitive, 5 = fully coherent). Reply with the single number."
)


def judge_coherence(
    texts: Sequence[str],
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> Optional[list[float]]:
    try:
        from openai import OpenAI
    except Exception:
        print("[judge] openai package not installed — skipping judge")
        return None
    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    scores = []
    for text in texts:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": RUBRIC},
                    {"role": "user", "content": text[:4000]},
                ],
                temperature=0.0,
                max_tokens=4,
            )
            raw = resp.choices[0].message.content.strip()
            scores.append(float(raw.split()[0].strip(".")))
        except Exception as exc:
            print(f"[judge] failed on one sample: {exc}")
            scores.append(float("nan"))
    return scores
