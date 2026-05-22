from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from .config import Settings
from .merger import format_timestamp, format_transcript_for_prompt


ANALYSIS_PROMPT = """<|system|>
You are an expert call center quality analyst with 15 years of experience coaching customer service agents.
You analyze call transcripts and provide specific, actionable feedback.
You MUST respond with only a valid JSON object. Do not include markdown, prose, or code fences.
<|end|>

<|user|>
Analyze this customer service call transcript and return a JSON object with exactly these fields:

{{
  "call_summary": "A 2-3 sentence overview of the call, the customer's issue, and the resolution.",
  "overall_sentiment": "positive | negative | neutral | mixed",
  "customer_sentiment_journey": "How the customer's mood changed through the call.",
  "agent_score": 1,
  "score_breakdown": {{
    "greeting_and_opening": 1,
    "active_listening": 1,
    "problem_resolution": 1,
    "professionalism": 1,
    "closing": 1
  }},
  "strengths": [
    "Specific strength with a direct quote or timestamp reference from the transcript"
  ],
  "improvement_areas": [
    "Specific area with concrete example and what the agent should do instead"
  ],
  "recommended_next_steps": [
    "Actionable coaching tip the agent can implement immediately"
  ],
  "key_moments": [
    {{
      "timestamp_range": "0:00-0:15",
      "speaker": "Agent | Customer",
      "description": "What happened and why it matters",
      "impact": "positive | negative | neutral"
    }}
  ],
  "compliance_flags": [
    "Any compliance concerns, or No compliance issues detected"
  ]
}}

Use integer scores from 1 to 10. Keep feedback tied to the transcript.

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


def parse_json_response(raw: str) -> dict[str, Any]:
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
        "error": "Failed to parse SLM output",
        "raw_output": raw,
    }


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

    for key in ("strengths", "improvement_areas", "recommended_next_steps", "key_moments", "compliance_flags"):
        value = normalized.get(key)
        if not isinstance(value, list):
            normalized[key] = [] if value in (None, "") else [str(value)]

    if not normalized["compliance_flags"]:
        normalized["compliance_flags"] = ["No compliance issues detected"]

    return normalized


def _clamp_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 5
    return max(1, min(10, score))


@lru_cache(maxsize=1)
def _load_llm(model_path: str, n_ctx: int, n_threads: int):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("llama-cpp-python is not installed.") from exc

    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=0,
        verbose=False,
    )


def analyze_transcript(
    transcript_blocks: list[dict],
    settings: Settings,
    analyzer_mode: str | None = None,
) -> dict[str, Any]:
    mode = (analyzer_mode or settings.analyzer_mode).lower()
    transcript = format_transcript_for_prompt(transcript_blocks, settings.max_transcript_chars)

    if mode not in {"auto", "llm", "heuristic"}:
        raise ValueError("analyzer_mode must be one of: auto, llm, heuristic")

    if mode in {"auto", "llm"} and settings.phi3_model_path.exists():
        prompt = ANALYSIS_PROMPT.format(transcript=transcript)
        llm = _load_llm(
            str(settings.phi3_model_path),
            settings.llama_ctx,
            settings.llama_threads,
        )
        response = llm(
            prompt,
            max_tokens=2048,
            temperature=0.2,
            top_p=0.9,
            stop=["<|end|>"],
        )
        raw_output = response["choices"][0]["text"].strip()
        analysis = normalize_analysis_schema(parse_json_response(raw_output))
        analysis["analysis_mode"] = "llm"
        analysis["model"] = settings.phi3_model_path.name
        return analysis

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

