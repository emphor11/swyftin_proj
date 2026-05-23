from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .merger import format_timestamp


def generate_reports(
    analysis: dict[str, Any],
    metadata: dict[str, Any],
    transcript_blocks: list[dict],
    output_dir: str | Path,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "report.json"
    md_path = output_path / "report.md"
    transcript_path = output_path / "transcript.txt"

    json_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown_report(analysis, metadata), encoding="utf-8")
    transcript_path.write_text(_build_transcript_text(transcript_blocks), encoding="utf-8")

    return {
        "json": json_path,
        "markdown": md_path,
        "transcript": transcript_path,
    }


def build_markdown_report(analysis: dict[str, Any], metadata: dict[str, Any]) -> str:
    generated_at = metadata.get("generated_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")
    filename = metadata.get("filename", "Transcript input")
    duration = metadata.get("duration_seconds")
    duration_text = format_timestamp(duration) if duration else "Unknown"

    score_breakdown = analysis.get("score_breakdown", {})
    key_moments = analysis.get("key_moments", [])
    compliance_flags = analysis.get("compliance_flags") or ["No compliance issues detected"]
    score_reasoning = _score_reasoning(analysis)

    return "\n".join(
        [
            "# Call Analysis Report",
            "",
            f"**Generated:** {generated_at}",
            f"**Audio File:** {filename}",
            f"**Duration:** {duration_text}",
            f"**Analysis Mode:** {analysis.get('analysis_mode', 'llm')}",
            "",
            "---",
            "",
            "## Call Summary",
            str(analysis.get("call_summary", "")),
            "",
            "## Sentiment",
            f"- **Overall:** {analysis.get('overall_sentiment', 'neutral')}",
            f"- **Customer Journey:** {analysis.get('customer_sentiment_journey', '')}",
            "",
            f"## Agent Score: {analysis.get('agent_score', 0)} / 10",
            *([score_reasoning, ""] if score_reasoning else [""]),
            "| Category | Score |",
            "|---|---:|",
            f"| Greeting & Opening | {score_breakdown.get('greeting_and_opening', 0)}/10 |",
            f"| Active Listening | {score_breakdown.get('active_listening', 0)}/10 |",
            f"| Problem Resolution | {score_breakdown.get('problem_resolution', 0)}/10 |",
            f"| Professionalism | {score_breakdown.get('professionalism', 0)}/10 |",
            f"| Closing | {score_breakdown.get('closing', 0)}/10 |",
            "",
            "## Strengths",
            _numbered_list(analysis.get("strengths", [])),
            "",
            "## Areas for Improvement",
            _numbered_list(analysis.get("improvement_areas", [])),
            "",
            "## Recommended Next Steps",
            _checklist(analysis.get("recommended_next_steps", [])),
            "",
            "## Key Moments",
            _moments_table(key_moments),
            "",
            "## Compliance Flags",
            _bullet_list(compliance_flags),
            "",
        ]
    )


def _score_reasoning(analysis: dict[str, Any]) -> str:
    reasoning = str(analysis.get("score_reasoning", "")).strip()
    if not reasoning:
        return ""
    return f"**Reasoning:** {reasoning}"


def _numbered_list(items: list[Any]) -> str:
    if not items:
        return "1. No items returned."
    return "\n".join(f"{idx}. {item}" for idx, item in enumerate(items, start=1))


def _bullet_list(items: list[Any]) -> str:
    if not items:
        return "- No items returned."
    return "\n".join(f"- {item}" for item in items)


def _checklist(items: list[Any]) -> str:
    if not items:
        return "- [ ] No next steps returned."
    return "\n".join(f"- [ ] {item}" for item in items)


def _moments_table(items: list[Any]) -> str:
    lines = ["| Time | Speaker | Event | Impact |", "|---|---|---|---|"]
    if not items:
        lines.append("| - | - | No key moments returned. | neutral |")
        return "\n".join(lines)

    for item in items:
        if not isinstance(item, dict):
            lines.append(f"| - | - | {str(item)} | neutral |")
            continue
        lines.append(
            "| {time} | {speaker} | {description} | {impact} |".format(
                time=str(item.get("timestamp_range", "-")).replace("|", "/"),
                speaker=str(item.get("speaker", "-")).replace("|", "/"),
                description=str(item.get("description", "")).replace("|", "/"),
                impact=str(item.get("impact", "neutral")).replace("|", "/"),
            )
        )
    return "\n".join(lines)


def _build_transcript_text(blocks: list[dict]) -> str:
    lines = []
    for block in blocks:
        start = format_timestamp(block.get("start", 0))
        end = format_timestamp(block.get("end", 0))
        lines.append(f"[{start}-{end}] {block.get('speaker', 'Speaker')}: {block.get('text', '')}")
    return "\n".join(lines) + "\n"
