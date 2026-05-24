from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from functools import lru_cache
from pathlib import Path

from .config import Settings
from .merger import fallback_diarization_from_whisper


class TranscriptionError(RuntimeError):
    pass


class DiarizationError(RuntimeError):
    pass


_WHISPER_TRANSCRIBE_LOCK = threading.Lock()


PYANNOTE_WORKER_CODE = r"""
import json
import os
import sys

try:
    import numpy as np

    if not hasattr(np, "NAN"):
        np.NAN = np.nan
except ImportError:
    pass

from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.environ.get("HF_TOKEN") or None,
)
if pipeline is None:
    raise RuntimeError("Could not load pyannote/speaker-diarization-3.1.")

num_speakers = int(sys.argv[2]) if len(sys.argv) > 2 else 2
diarization_kwargs = {}
if num_speakers > 0:
    diarization_kwargs["num_speakers"] = num_speakers

segments = []
for turn, _, speaker in pipeline(sys.argv[1], **diarization_kwargs).itertracks(yield_label=True):
    segments.append(
        {
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        }
    )

print("__PYANNOTE_SEGMENTS__" + json.dumps(segments))
"""


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
    with _WHISPER_TRANSCRIBE_LOCK:
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


def diarize(audio_path: str | Path, settings: Settings) -> list[dict]:
    env = os.environ.copy()
    if settings.hf_token:
        env["HF_TOKEN"] = settings.hf_token

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                PYANNOTE_WORKER_CODE,
                str(audio_path),
                str(settings.pyannote_num_speakers),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=settings.pyannote_timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise DiarizationError(
            f"Pyannote diarization timed out after {settings.pyannote_timeout_seconds}s."
        ) from exc

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise DiarizationError(message or "Pyannote diarization failed.")

    for line in result.stdout.splitlines():
        if line.startswith("__PYANNOTE_SEGMENTS__"):
            return json.loads(line.removeprefix("__PYANNOTE_SEGMENTS__"))

    raise DiarizationError("Pyannote completed but did not return speaker segments.")


def diarize_with_fallback(
    audio_path: str | Path,
    whisper_segments: list[dict],
    settings: Settings,
) -> tuple[list[dict], str | None]:
    try:
        return diarize(audio_path, settings), None
    except Exception as exc:  # noqa: BLE001 - fallback should catch setup/runtime failures.
        return fallback_diarization_from_whisper(whisper_segments), str(exc)
