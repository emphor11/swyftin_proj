from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


AGENT_CUES = {
    "thank you for calling": 5,
    "thanks for calling": 5,
    "how may i help": 5,
    "how can i help": 5,
    "support": 3,
    "account verification": 4,
    "verify your": 4,
    "calling you": 3,
    "help you": 3,
    "guide you": 3,
    "please open": 3,
    "open any desk": 4,
    "give me the any desk": 4,
    "finance department": 3,
    "withdrawal": 2,
    "platform": 2,
}

CUSTOMER_CUES = {
    "my name is": 2,
    "i need help": 4,
    "i'm having": 4,
    "i am having": 4,
    "i can't": 4,
    "i cannot": 4,
    "my account": 3,
    "my order": 3,
    "my bank": 2,
    "yes": 1,
}


def _normalize_text_for_matching(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("’", "'")).strip()


def _contains_cue(text: str, cue: str) -> bool:
    tokens = [re.escape(token) for token in cue.split()]
    if not tokens:
        return False
    pattern = r"(?<!\w)" + r"\s+".join(tokens) + r"(?!\w)"
    return re.search(pattern, text) is not None


def _score_cues(text: str, cues: dict[str, int]) -> int:
    normalized = _normalize_text_for_matching(text)
    return sum(weight for cue, weight in cues.items() if _contains_cue(normalized, cue))


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

    scores: dict[str, int] = {}
    for speaker in speaker_order:
        sample = speaker_text.get(speaker, "")
        agent_score = _score_cues(sample, AGENT_CUES)
        customer_score = _score_cues(sample, CUSTOMER_CUES)
        scores[speaker] = agent_score - customer_score

    if scores and max(scores.values()) > 0:
        agent_speaker = max(speaker_order, key=lambda speaker: scores.get(speaker, 0))
    else:
        agent_speaker = speaker_order[0] if speaker_order else "SPEAKER_00"

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


def _should_flip_fallback_speaker(previous: dict, current: dict) -> bool:
    previous_end = float(previous.get("end", previous.get("start", 0.0)))
    current_start = float(current.get("start", previous_end))
    gap = current_start - previous_end
    previous_text = str(previous.get("text", "")).strip()
    current_text = str(current.get("text", "")).strip()
    current_words = re.findall(r"\b\w+\b", current_text)
    if previous_text.endswith("?") and len(current_words) > 1:
        return True
    if gap <= 0.5:
        return False
    if gap >= 2.0:
        return True

    return previous_text.endswith("?") or (gap > 0.8 and len(current_words) <= 3)


def fallback_diarization_from_whisper(whisper_segments: list[dict]) -> list[dict]:
    diarized: list[dict] = []
    current_speaker = "SPEAKER_00"

    for index, segment in enumerate(whisper_segments):
        if index > 0 and _should_flip_fallback_speaker(whisper_segments[index - 1], segment):
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
        separator = "\n\n[Transcript truncated for context length]\n\n"
        half = max(1, (max_chars - len(separator)) // 2)

        head: list[str] = []
        head_length = 0
        for line in lines:
            next_length = len(line) + (1 if head else 0)
            if head and head_length + next_length > half:
                break
            if not head and len(line) > half:
                head.append(line[:half].rstrip())
                break
            head.append(line)
            head_length += next_length

        tail: list[str] = []
        tail_length = 0
        for line in reversed(lines):
            next_length = len(line) + (1 if tail else 0)
            if tail and tail_length + next_length > half:
                break
            if not tail and len(line) > half:
                tail.append(line[-half:].lstrip())
                break
            tail.append(line)
            tail_length += next_length

        transcript = (
            "\n".join(head).rstrip()
            + separator
            + "\n".join(reversed(tail)).lstrip()
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

    def normalize_explicit_label(label: str) -> str:
        normalized = re.sub(r"\s+", " ", label.strip())
        lowered = normalized.lower()
        if lowered == "agent":
            return "Agent"
        if lowered == "customer":
            return "Customer"
        if re.fullmatch(r"speaker\s+1", lowered):
            return "Agent"
        if re.fullmatch(r"speaker\s+2", lowered):
            return "Customer"
        return normalized.upper() if lowered.startswith("speaker_") else normalized

    cursor = 0.0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.match(line)
        if match:
            start = timestamp_to_seconds(match.group("start")) if match.group("start") else cursor
            end = timestamp_to_seconds(match.group("end")) if match.group("end") else start + 5
            source_speaker = match.group("speaker").strip()
            speaker = normalize_explicit_label(source_speaker)
            blocks.append(
                {
                    "speaker": speaker,
                    "source_speaker": source_speaker,
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

    role_map = _infer_role_map(blocks)
    for block in blocks:
        speaker = str(block.get("speaker", ""))
        if speaker not in {"Agent", "Customer"}:
            block["speaker"] = role_map.get(speaker, speaker)

    return blocks


def write_transcript(blocks: list[dict], path: Path) -> None:
    path.write_text(format_transcript_for_prompt(blocks) + "\n", encoding="utf-8")
