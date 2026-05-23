from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from .config import Settings
from .merger import format_timestamp


SUMMARY_PROMPT = """<|system|>
You are a call center quality analyst. Respond with ONLY valid JSON. No markdown, no explanation.
<|end|>
<|user|>
Analyze this call transcript. Return JSON with these exact keys:
- "call_summary": 2-3 sentence overview
- "overall_sentiment": one of positive/negative/neutral/mixed
- "customer_sentiment_journey": e.g. "frustrated -> satisfied"
- "agent_score": integer 1-10
- "score_reasoning": one sentence explaining the score

TRANSCRIPT:
{transcript}
<|end|>

<|assistant|>
"""

COACHING_PROMPT = """<|system|>
You are a call center quality analyst. Respond with ONLY valid JSON. No markdown, no explanation.
<|end|>
<|user|>
This call was scored {agent_score}/10. Summary: {call_summary}

Based on the transcript below, return JSON with these exact keys:
- "strengths": array of 2-3 specific strengths with references to what the agent said or did
- "improvement_areas": array of 2-3 specific weaknesses with what to do differently
- "recommended_next_steps": array of 2-3 actionable coaching tips

TRANSCRIPT:
{transcript}
<|end|>
<|assistant|>
"""

MOMENTS_PROMPT = """<|system|>
You are a call center quality analyst. Respond with ONLY valid JSON. No markdown, no explanation.
<|end|>
<|user|>
This call was scored {agent_score}/10.

Based on the transcript below, return JSON with these exact keys:
- "score_breakdown": object with keys greeting_and_opening, active_listening, problem_resolution, professionalism, closing (each integer 1-10)
- "key_moments": array of 1-3 objects with keys timestamp_range, speaker, description, impact
- "compliance_flags": array of strings (empty array if no issues)

TRANSCRIPT:
{transcript}
<|end|>
<|assistant|>
"""


DEFAULT_BREAKDOWN = {
    "greeting_and_opening": 5,
    "active_listening": 5,
    "problem_resolution": 5,
    "professionalism": 5,
    "closing": 5,
}


def _parse_json(raw: str) -> dict[str, Any]:
    candidates = [raw.strip()]
    candidates.append(re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE).replace("```", "").strip())

    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        candidates.append(match.group(0).strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return {
        "error": "parse_failed",
        "raw": raw,
    }


def parse_json_response(raw: str) -> dict[str, Any]:
    return _parse_json(raw)


def _format_transcript(blocks: list[dict], max_chars: int = 3000) -> str:
    lines = []
    for block in blocks:
        speaker = str(block.get("speaker", "Speaker")).strip() or "Speaker"
        text = str(block.get("text", "")).strip()
        if text:
            lines.append(f"[{speaker}]: {text}")

    transcript = "\n".join(lines)
    if len(transcript) <= max_chars:
        return transcript

    marker = "\n[transcript truncated]"
    return transcript[: max_chars - len(marker)].rstrip() + marker


def normalize_analysis_schema(analysis: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(analysis)

    normalized.setdefault("call_summary", "The call was analyzed, but the summary was not returned.")
    normalized.setdefault("overall_sentiment", "neutral")
    normalized.setdefault("customer_sentiment_journey", "Not enough information to determine a journey.")
    normalized.setdefault("agent_score", 5)
    normalized.setdefault("score_breakdown", {})
    normalized["score_breakdown"] = {
        **DEFAULT_BREAKDOWN,
        **{
            key: _clamp_score(value)
            for key, value in dict(normalized.get("score_breakdown", {})).items()
        },
    }
    normalized["agent_score"] = _clamp_score(normalized.get("agent_score", 5))

    normalized.setdefault("compliance_flags", [])

    for key in ("strengths", "improvement_areas", "recommended_next_steps", "key_moments", "compliance_flags"):
        value = normalized.get(key)
        if not isinstance(value, list):
            normalized[key] = [] if value in (None, "") else [str(value)]

    return normalized


def _clamp_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 5
    return max(1, min(10, score))


@lru_cache(maxsize=1)
def _load_llm(model_path: str):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("llama-cpp-python is not installed.") from exc

    return Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=4,
        n_gpu_layers=0,
        verbose=False,
    )


def _run_llm_json(llm: Any, prompt: str) -> dict[str, Any]:
    response = llm(
        prompt,
        max_tokens=400,
        temperature=0.3,
        stop=["<|end|>"],
    )
    raw_output = response["choices"][0]["text"].strip()
    return _parse_json(raw_output)


def _analyze_with_llm(transcript_blocks: list[dict], settings: Settings) -> dict[str, Any]:
    transcript = _format_transcript(transcript_blocks, max_chars=3000)
    llm = _load_llm(str(settings.phi3_model_path))

    summary = _run_llm_json(llm, SUMMARY_PROMPT.format(transcript=transcript))
    agent_score = _clamp_score(summary.get("agent_score", 5))
    call_summary = str(summary.get("call_summary", "Summary unavailable."))

    coaching = _run_llm_json(
        llm,
        COACHING_PROMPT.format(
            transcript=transcript,
            agent_score=agent_score,
            call_summary=call_summary,
        ),
    )
    moments = _run_llm_json(
        llm,
        MOMENTS_PROMPT.format(
            transcript=transcript,
            agent_score=agent_score,
        ),
    )

    analysis: dict[str, Any] = {}
    errors = {}
    for name, result in (("summary", summary), ("coaching", coaching), ("moments", moments)):
        if "error" in result:
            errors[name] = result
            continue
        analysis.update(result)

    if errors:
        analysis["llm_errors"] = errors

    analysis = normalize_analysis_schema(analysis)
    analysis["analysis_mode"] = "llm"
    analysis["model"] = settings.phi3_model_path.name
    return analysis


def analyze_transcript(
    transcript_blocks: list[dict],
    settings: Settings,
    analyzer_mode: str | None = None,
) -> dict[str, Any]:
    mode = (analyzer_mode or settings.analyzer_mode).lower()

    if mode not in {"auto", "llm", "heuristic"}:
        raise ValueError("analyzer_mode must be one of: auto, llm, heuristic")

    if mode in {"auto", "llm"} and settings.phi3_model_path.exists():
        return _analyze_with_llm(transcript_blocks, settings)

    if mode == "llm":
        raise FileNotFoundError(
            f"Phi-3 model file not found at {settings.phi3_model_path}. "
            "Download the GGUF file or use --analyzer-mode heuristic for a smoke test."
        )

    analysis = heuristic_analysis(transcript_blocks)
    analysis["analysis_mode"] = "heuristic-dev-fallback"
    analysis["model"] = "rule-based fallback"
    return analysis


def heuristic_analysis(blocks: list[dict]) -> dict[str, Any]:
    joined = " ".join(str(block.get("text", "")) for block in blocks).lower()
    agent_text = " ".join(str(block.get("text", "")) for block in blocks if block.get("speaker") == "Agent").lower()
    customer_text = " ".join(str(block.get("text", "")) for block in blocks if block.get("speaker") == "Customer").lower()

    negative_words = ("frustrated", "angry", "upset", "problem", "issue", "wrong", "not working", "charged", "cancel")
    positive_words = ("thank", "thanks", "great", "resolved", "helpful", "appreciate", "perfect")

    negative_count = sum(joined.count(word) for word in negative_words)
    positive_count = sum(joined.count(word) for word in positive_words)

    sentiment = "mixed"
    if positive_count > negative_count + 1:
        sentiment = "positive"
    elif negative_count > positive_count + 1:
        sentiment = "negative"
    elif positive_count == 0 and negative_count == 0:
        sentiment = "neutral"

    greeting_score = 8 if any(cue in agent_text for cue in ("thank you for calling", "how can i help", "my name is")) else 5
    listening_score = 8 if any(cue in agent_text for cue in ("understand", "i see", "let me check", "sorry")) else 5
    resolution_score = 8 if any(cue in agent_text for cue in ("resolved", "fixed", "refund", "credit", "next step")) else 5
    professionalism_score = 8 if any(cue in agent_text for cue in ("please", "thank", "sorry", "happy to help")) else 6
    closing_score = 8 if any(cue in agent_text for cue in ("anything else", "have a great", "thank you for calling")) else 5

    score_breakdown = {
        "greeting_and_opening": greeting_score,
        "active_listening": listening_score,
        "problem_resolution": resolution_score,
        "professionalism": professionalism_score,
        "closing": closing_score,
    }
    agent_score = round(sum(score_breakdown.values()) / len(score_breakdown))

    first_customer = next((b for b in blocks if b.get("speaker") == "Customer"), None)
    last_customer = next((b for b in reversed(blocks) if b.get("speaker") == "Customer"), None)
    first_agent = next((b for b in blocks if b.get("speaker") == "Agent"), None)

    summary_issue = _shorten(first_customer.get("text", "") if first_customer else "The customer described their issue.")
    summary_resolution = _shorten(last_customer.get("text", "") if last_customer else "The call ended without enough detail.")

    return normalize_analysis_schema(
        {
            "call_summary": (
                f"The customer contacted support about: {summary_issue} "
                f"By the end of the call, the customer response was: {summary_resolution}"
            ),
            "overall_sentiment": sentiment,
            "customer_sentiment_journey": _sentiment_journey(sentiment, customer_text),
            "agent_score": agent_score,
            "score_breakdown": score_breakdown,
            "strengths": _fallback_strengths(first_agent, listening_score, resolution_score),
            "improvement_areas": _fallback_improvements(score_breakdown),
            "recommended_next_steps": [
                "Use a clear ownership statement early in the call, such as 'I can help with that today.'",
                "Summarize the customer's issue before moving into troubleshooting or policy details.",
                "Close with the resolution, expected timeline, and one final check for remaining questions.",
            ],
            "key_moments": _fallback_key_moments(blocks),
            "compliance_flags": _fallback_compliance_flags(agent_text),
        }
    )


def _shorten(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _sentiment_journey(sentiment: str, customer_text: str) -> str:
    if "thank" in customer_text or "appreciate" in customer_text:
        return "Concerned at the start, then more reassured by the end."
    if sentiment == "negative":
        return "The customer remained frustrated or concerned during the call."
    if sentiment == "positive":
        return "The customer sounded positive or satisfied overall."
    return "The customer's mood appears mostly neutral with limited emotional change."


def _fallback_strengths(first_agent: dict | None, listening_score: int, resolution_score: int) -> list[str]:
    strengths = []
    if first_agent:
        strengths.append(
            f"Opened the interaction with a clear agent turn around {format_timestamp(first_agent.get('start', 0))}: "
            f"'{_shorten(str(first_agent.get('text', '')), 120)}'"
        )
    if listening_score >= 8:
        strengths.append("Used acknowledgement language that signals active listening.")
    if resolution_score >= 8:
        strengths.append("Provided a resolution-oriented response instead of leaving the issue open-ended.")
    return strengths or ["Maintained a usable call flow with enough information for follow-up coaching."]


def _fallback_improvements(score_breakdown: dict[str, int]) -> list[str]:
    improvements = []
    if score_breakdown["greeting_and_opening"] < 7:
        improvements.append("Strengthen the opening with name, company, and a direct offer to help.")
    if score_breakdown["active_listening"] < 7:
        improvements.append("Add a brief recap of the customer's issue to prove understanding before solving.")
    if score_breakdown["problem_resolution"] < 7:
        improvements.append("State the resolution or next action more explicitly, including owner and timeline.")
    if score_breakdown["closing"] < 7:
        improvements.append("End with a concise recap and ask whether the customer needs anything else.")
    return improvements or ["Continue making feedback specific by tying each coaching point to a timestamp."]


def _fallback_key_moments(blocks: list[dict]) -> list[dict]:
    moments = []
    for block in blocks[:3]:
        start = format_timestamp(block.get("start", 0))
        end = format_timestamp(block.get("end", 0))
        moments.append(
            {
                "timestamp_range": f"{start}-{end}",
                "speaker": block.get("speaker", "Speaker"),
                "description": _shorten(str(block.get("text", "")), 140),
                "impact": "neutral",
            }
        )
    return moments


def _fallback_compliance_flags(agent_text: str) -> list[str]:
    if any(cue in agent_text for cue in ("verify", "account number", "security")):
        return ["No compliance issues detected"]
    return ["Account verification was not clearly present in the transcript; verify whether this was required for the call type."]
