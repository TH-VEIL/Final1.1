"""VEIL GUARD: review a redacted prompt and model response locally via Gemini."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class GuardReview:
    verdict: str
    score: int
    reason: str
    feedback: str
    reveal_placeholders: list[str] = field(default_factory=list)
    retry_instruction: str = ""


def ollama_generate(prompt: str, model_name: str, system_instruction: str = "", timeout: int = 60) -> str:
    """Call a local Ollama server; no prompt or response leaves the machine."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "system": system_instruction,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 512},
    }
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("error", str(exc))
        except Exception:
            detail = str(exc)
        raise RuntimeError(f"Ollama request failed: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Ollama is not reachable at http://127.0.0.1:11434. "
            "Start it with `ollama serve` and make sure the model is installed."
        ) from exc
    if body.get("error"):
        raise RuntimeError(f"Ollama error: {body['error']}")
    answer = body.get("response", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty response.")
    return answer


class VeilGuard:
    """Scores one answer and decides whether one targeted reveal is justified."""

    def __init__(self, client, model_name: str):
        self.client = client
        self.model_name = model_name

    def review(self, redacted_prompt: str, answer: str, placeholders: list[str]) -> GuardReview:
        from google.genai import types

        review_prompt = build_review_prompt(redacted_prompt, answer, placeholders)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=review_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=300,
                response_mime_type="application/json",
            ),
        )
        raw = getattr(response, "text", "") or ""
        review = self._parse(raw)
        return self._enforce(review, answer, placeholders)

    @staticmethod
    def _parse(raw: str) -> GuardReview:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        try:
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                # Reasoning models such as DeepSeek-R1 may wrap the JSON in
                # <think>...</think> or add a short explanation first.
                match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
                if not match:
                    raise
                data = json.loads(match.group(0))
            verdict = str(data.get("verdict", "BAD_PROMPT")).upper()
            if verdict not in {"APPROVE", "RETRY_WITH_REVEAL", "BAD_PROMPT"}:
                verdict = "BAD_PROMPT"
            return GuardReview(
                verdict=verdict,
                score=max(0, min(100, int(data.get("score", 0)))),
                reason=str(data.get("reason", "UNPARSEABLE_REVIEW")),
                feedback=str(data.get("feedback", "VEIL GUARD could not explain the decision.")),
                reveal_placeholders=list(data.get("reveal_placeholders") or []),
                retry_instruction=str(data.get("retry_instruction", "")),
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            return GuardReview(
                verdict="BAD_PROMPT",
                score=0,
                reason="GUARD_PARSE_ERROR",
                feedback="VEIL GUARD returned an invalid review, so the answer was not released.",
            )

    @staticmethod
    def _enforce(review: GuardReview, answer: str, placeholders: list[str]) -> GuardReview:
        allowed = set(placeholders)
        unknown_tokens = re.findall(r"(?:<[A-Z][A-Z0-9_]*_\d+>|\[[A-Z][A-Z0-9_]*_\d+\])", answer)
        unknown = set(unknown_tokens) - allowed - {f"[{item[1:-1]}]" for item in allowed}
        requested = [item for item in review.reveal_placeholders if item in allowed]
        if unknown:
            return GuardReview(
                verdict="BAD_PROMPT",
                score=0,
                reason="UNKNOWN_PLACEHOLDER",
                feedback=f"The answer introduced unknown placeholder(s): {', '.join(sorted(unknown))}.",
            )
        if review.verdict == "RETRY_WITH_REVEAL" and not requested:
            return GuardReview(
                verdict="BAD_PROMPT",
                score=review.score,
                reason="NO_JUSTIFIED_REVEAL",
                feedback="The guard did not identify a specific sensitive value that would improve the answer.",
            )
        review.reveal_placeholders = requested
        return review


def build_review_prompt(redacted_prompt: str, answer: str, placeholders: list[str]) -> str:
    expected = ", ".join(placeholders) if placeholders else "(none)"
    return f"""
You are VEIL GUARD, a strict answer-quality and privacy gate.

Review the redacted user prompt and the LLM answer below. Score the answer from
0 to 100 using task completion, correctness, clarity, and placeholder safety.
Do not guess hidden values. You may recommend revealing a placeholder only when
the actual value is genuinely necessary to complete the user's requested task,
not merely because it would be convenient. Request the smallest possible set.

Return ONLY valid JSON with exactly this shape:
{{
  "verdict": "APPROVE" | "RETRY_WITH_REVEAL" | "BAD_PROMPT",
  "score": 0,
  "reason": "short label",
  "feedback": "brief explanation",
  "reveal_placeholders": [],
  "retry_instruction": "instruction for the second attempt or empty string"
}}

Rules:
- APPROVE means the answer is satisfactory and safe to return after local restoration.
- RETRY_WITH_REVEAL means a specific hidden value is genuinely required and a
  second answer could be improved by revealing only listed placeholders.
- BAD_PROMPT means the answer is not satisfactory and revealing data would not
  solve the problem, or the request is ambiguous/unsafe.
- reveal_placeholders must contain only values from the allowed list.
- Never include or infer the original values.

Allowed placeholders: {expected}

REDACTED USER PROMPT:
{redacted_prompt}

LLM ANSWER:
{answer}
"""


class OllamaVeilGuard(VeilGuard):
    """Local equivalent of VEIL GUARD using the same Ollama model/API."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def review(self, redacted_prompt: str, answer: str, placeholders: list[str]) -> GuardReview:
        raw = ollama_generate(
            build_review_prompt(redacted_prompt, answer, placeholders),
            self.model_name,
            system_instruction="Return only the JSON object requested by the VEIL GUARD prompt.",
        )
        return self._enforce(self._parse(raw), answer, placeholders)
