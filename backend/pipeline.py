from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .analyzer import analyze_transcript
from .config import Settings, get_settings
from .merger import merge_transcript, parse_labeled_transcript
from .report_generator import generate_reports
from .transcriber import (
    diarize_with_fallback,
    get_audio_duration,
    normalize_audio,
    transcribe,
)


ProgressCallback = Callable[[str, int, str], None]


def _emit(callback: ProgressCallback | None, stage: str, progress: int, message: str) -> None:
    if callback:
        callback(stage, progress, message)


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    analyzer_mode: str | None = None,
    settings: Settings | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    input_path = Path(input_path)
    job_id = uuid.uuid4().hex[:12]
    report_dir = Path(output_dir) if output_dir else settings.output_dir / job_id
    report_dir.mkdir(parents=True, exist_ok=True)

    _emit(progress_callback, "normalizing", 10, "Normalizing audio to 16 kHz mono WAV.")
    normalized_path = normalize_audio(input_path, settings)
    duration = get_audio_duration(normalized_path)

    _emit(progress_callback, "transcribing", 30, "Transcribing speech with open-source Whisper.")
    whisper_segments = transcribe(normalized_path, settings)

    _emit(progress_callback, "diarizing", 50, "Assigning speaker turns.")
    diarization_segments, diarization_warning = diarize_with_fallback(
        normalized_path,
        whisper_segments,
        settings,
    )

    _emit(progress_callback, "merging", 65, "Aligning transcript text with speaker labels.")
    transcript_blocks = merge_transcript(whisper_segments, diarization_segments)

    _emit(progress_callback, "analyzing", 82, "Generating coaching feedback with the SLM.")
    analysis = analyze_transcript(transcript_blocks, settings, analyzer_mode)
    if diarization_warning:
        analysis.setdefault("pipeline_warnings", []).append(
            f"Pyannote diarization fallback used: {diarization_warning}"
        )

    metadata = {
        "id": report_dir.name,
        "filename": input_path.name,
        "duration_seconds": duration,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    _emit(progress_callback, "generating_report", 95, "Writing JSON, Markdown, and transcript outputs.")
    paths = generate_reports(analysis, metadata, transcript_blocks, report_dir)

    shutil.copy2(input_path, report_dir / input_path.name)
    _emit(progress_callback, "complete", 100, "Analysis complete.")

    return {
        "id": report_dir.name,
        "metadata": metadata,
        "analysis": analysis,
        "transcript": transcript_blocks,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def run_transcript_pipeline(
    transcript_path: str | Path,
    output_dir: str | Path | None = None,
    analyzer_mode: str | None = None,
    settings: Settings | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    transcript_path = Path(transcript_path)
    job_id = uuid.uuid4().hex[:12]
    report_dir = Path(output_dir) if output_dir else settings.output_dir / job_id
    report_dir.mkdir(parents=True, exist_ok=True)

    _emit(progress_callback, "loading_transcript", 20, "Loading labeled transcript fixture.")
    transcript_blocks = parse_labeled_transcript(transcript_path.read_text(encoding="utf-8"))

    _emit(progress_callback, "analyzing", 75, "Generating coaching feedback.")
    analysis = analyze_transcript(transcript_blocks, settings, analyzer_mode)
    analysis.setdefault("pipeline_warnings", []).append(
        "Transcript-only mode was used for development smoke testing; audio transcription was skipped."
    )

    metadata = {
        "id": report_dir.name,
        "filename": transcript_path.name,
        "duration_seconds": transcript_blocks[-1]["end"] if transcript_blocks else 0,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    _emit(progress_callback, "generating_report", 95, "Writing JSON, Markdown, and transcript outputs.")
    paths = generate_reports(analysis, metadata, transcript_blocks, report_dir)
    _emit(progress_callback, "complete", 100, "Analysis complete.")

    return {
        "id": report_dir.name,
        "metadata": metadata,
        "analysis": analysis,
        "transcript": transcript_blocks,
        "paths": {key: str(value) for key, value in paths.items()},
    }
