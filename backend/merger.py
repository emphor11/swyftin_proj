from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


AGENT_CUES = (
    "thank you for calling",
    "thanks for calling",
    "how may i help",
    "how can i help",
    "my name is",
    "this is ",
    "support",
    "account verification",
    "verify your",
)


def format_timestamp(seconds: float | int | None) -> str:
    if seconds is None:
        return "0:00"
    seconds = max(0, int(round(float(seconds))))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def timestamp_to_seconds(value: str | None) -> float:
    if not value:
        return 0.0
    parts = [p.strip() for p in value.split(":")]
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return 0.0
    return 0.0


def segment_overlap(
    a_start: float,
    a_end: float,
    b_start: float,
    b_end: float,
) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _choose_speaker_for_segment(
    whisper_segment: dict,
    diarization_segments: list[dict],
) -> str:
    start = float(whisper_segment.get("start", 0.0))
    end = float(whisper_segment.get("end", start))
    best_speaker = "SPEAKER_00"
    best_overlap = 0.0

    for diarized in diarization_segments:
        overlap = segment_overlap(
            start,
            end,
            float(diarized.get("start", 0.0)),
            float(diarized.get("end", 0.0)),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = str(diarized.get("speaker", best_speaker))

    if best_overlap > 0:
        return best_speaker

    midpoint = (start + end) / 2
    closest = min(
        diarization_segments,
        key=lambda item: abs(
            midpoint
            - (
                float(item.get("start", 0.0))
                + float(item.get("end", float(item.get("start", 0.0))))
            )
            / 2
        ),
        default={"speaker": best_speaker},
    )
    return str(closest.get("speaker", best_speaker))


def _infer_role_map(labeled_segments: list[dict]) -> dict[str, str]:
    speaker_order: list[str] = []
    speaker_text: dict[str, str] = {}

    for segment in labeled_segments:
        speaker = str(segment["speaker"])
        if speaker not in speaker_order:
            speaker_order.append(speaker)
        speaker_text[speaker] = f"{speaker_text.get(speaker, '')} {segment.get('text', '')}"

    agent_speaker = speaker_order[0] if speaker_order else "SPEAKER_00"
    for speaker in speaker_order:
        sample = speaker_text.get(speaker, "").lower()
        if any(cue in sample for cue in AGENT_CUES):
            agent_speaker = speaker
            break

    role_map = {}
    for speaker in speaker_order:
        role_map[speaker] = "Agent" if speaker == agent_speaker else "Customer"
    return role_map


def merge_transcript(
    whisper_segments: list[dict],
    diarization_segments: list[dict],
    merge_gap_seconds: float = 1.0,
) -> list[dict]:
    if not whisper_segments:
        return []

    if not diarization_segments:
        diarization_segments = fallback_diarization_from_whisper(whisper_segments)

    raw_segments: list[dict] = []
    for item in whisper_segments:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        raw_segments.append(
            {
                "speaker": _choose_speaker_for_segment(item, diarization_segments),
                "start": float(item.get("start", 0.0)),
                "end": float(item.get("end", item.get("start", 0.0))),
                "text": text,
            }
        )

    role_map = _infer_role_map(raw_segments)
    merged: list[dict] = []

    for item in raw_segments:
        role = role_map.get(item["speaker"], item["speaker"])
        if (
            merged
            and merged[-1]["speaker"] == role
            and item["start"] - merged[-1]["end"] <= merge_gap_seconds
        ):
            merged[-1]["end"] = item["end"]
            merged[-1]["text"] = f"{merged[-1]['text']} {item['text']}".strip()
            continue

        merged.append(
            {
                "speaker": role,
                "source_speaker": item["speaker"],
                "start": item["start"],
                "end": item["end"],
                "text": item["text"],
            }
        )

    return merged


def fallback_diarization_from_whisper(whisper_segments: list[dict]) -> list[dict]:
    diarized: list[dict] = []
    current_speaker = "SPEAKER_00"

    for index, segment in enumerate(whisper_segments):
        if index > 0:
            current_speaker = "SPEAKER_01" if current_speaker == "SPEAKER_00" else "SPEAKER_00"
        diarized.append(
            {
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", segment.get("start", 0.0))),
                "speaker": current_speaker,
            }
        )

    return diarized


def format_transcript_for_prompt(blocks: Iterable[dict], max_chars: int | None = None) -> str:
    lines = []
    for block in blocks:
        start = format_timestamp(block.get("start", 0.0))
        end = format_timestamp(block.get("end", 0.0))
        speaker = block.get("speaker", "Speaker")
        text = str(block.get("text", "")).strip()
        if text:
            lines.append(f"[{start}-{end}] {speaker}: {text}")

    transcript = "\n".join(lines)
    if max_chars and len(transcript) > max_chars:
        half = max_chars // 2
        transcript = (
            transcript[:half].rstrip()
            + "\n\n[Transcript truncated for context length]\n\n"
            + transcript[-half:].lstrip()
        )
    return transcript


def parse_labeled_transcript(text: str) -> list[dict]:
    blocks: list[dict] = []
    pattern = re.compile(
        r"^(?:(?:\[)?(?P<start>\d{1,2}:\d{2}(?::\d{2})?)"
        r"(?:\s*-\s*(?P<end>\d{1,2}:\d{2}(?::\d{2})?))?(?:\])?\s*)?"
        r"(?P<speaker>Agent|Customer|SPEAKER_\d+|Speaker\s+\d+)\s*:\s*(?P<text>.+)$",
        re.IGNORECASE,
    )

    cursor = 0.0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if match:
            start = timestamp_to_seconds(match.group("start")) if match.group("start") else cursor
            end = timestamp_to_seconds(match.group("end")) if match.group("end") else start + 5
            speaker = match.group("speaker").replace("Speaker 1", "Agent").replace("Speaker 2", "Customer")
            speaker = "Agent" if speaker.lower() == "agent" else "Customer" if speaker.lower() == "customer" else speaker
            blocks.append(
                {
                    "speaker": speaker,
                    "source_speaker": speaker,
                    "start": start,
                    "end": end,
                    "text": match.group("text").strip(),
                }
            )
            cursor = end
            continue

        speaker = "Customer" if blocks and blocks[-1]["speaker"] == "Agent" else "Agent"
        blocks.append(
            {
                "speaker": speaker,
                "source_speaker": speaker,
                "start": cursor,
                "end": cursor + 5,
                "text": line,
            }
        )
        cursor += 5

    return blocks


def write_transcript(blocks: list[dict], path: Path) -> None:
    path.write_text(format_transcript_for_prompt(blocks) + "\n", encoding="utf-8")

