from __future__ import annotations

import subprocess
import uuid
from functools import lru_cache
from pathlib import Path

from .config import Settings
from .merger import fallback_diarization_from_whisper


class TranscriptionError(RuntimeError):
    pass


class DiarizationError(RuntimeError):
    pass


def get_audio_duration(audio_path: str | Path) -> float:
    path = Path(audio_path)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def normalize_audio(input_path: str | Path, settings: Settings) -> Path:
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Audio file not found: {source}")

    output_path = settings.temp_dir / f"{uuid.uuid4().hex}_16khz_mono.wav"
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise TranscriptionError(result.stderr.strip() or "ffmpeg failed to normalize audio")
    return output_path


@lru_cache(maxsize=2)
def _load_whisper_model(model_name: str):
    try:
        import whisper
    except ImportError as exc:
        raise TranscriptionError(
            "openai-whisper is not installed. Install backend/requirements.txt first."
        ) from exc
    return whisper.load_model(model_name)


def transcribe(audio_path: str | Path, settings: Settings) -> list[dict]:
    model = _load_whisper_model(settings.whisper_model)
    kwargs = {}
    if settings.whisper_language:
        kwargs["language"] = settings.whisper_language
    result = model.transcribe(str(audio_path), **kwargs)
    segments = []
    for item in result.get("segments", []):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            {
                "start": float(item.get("start", 0.0)),
                "end": float(item.get("end", item.get("start", 0.0))),
                "text": text,
            }
        )
    return segments


@lru_cache(maxsize=1)
def _load_pyannote_pipeline(token: str | None):
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationError(
            "pyannote.audio is not installed. Falling back to turn alternation."
        ) from exc

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )
    if pipeline is None:
        raise DiarizationError(
            "Could not load pyannote/speaker-diarization-3.1. Check HF_TOKEN and model terms."
        )
    return pipeline


def diarize(audio_path: str | Path, settings: Settings) -> list[dict]:
    pipeline = _load_pyannote_pipeline(settings.hf_token)
    diarization = pipeline(str(audio_path))
    segments: list[dict] = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )

    return segments


def diarize_with_fallback(
    audio_path: str | Path,
    whisper_segments: list[dict],
    settings: Settings,
) -> tuple[list[dict], str | None]:
    try:
        return diarize(audio_path, settings), None
    except Exception as exc:  # noqa: BLE001 - fallback should catch setup/runtime failures.
        return fallback_diarization_from_whisper(whisper_segments), str(exc)

