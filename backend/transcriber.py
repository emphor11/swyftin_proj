from __future__ import annotations

import atexit
from collections import deque
import json
import os
import queue
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


class PyannoteWorkerManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._proc: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[dict] = queue.Queue()
        self._stderr_lines: deque[str] = deque(maxlen=50)
        self._ready = False
        self._unavailable_reason: str | None = None

    def warmup(self, settings: Settings) -> None:
        with self._lock:
            self._unavailable_reason = None
            self._ensure_worker_locked(settings)

    def mark_unavailable(self, reason: str) -> None:
        with self._lock:
            self._unavailable_reason = reason
            self._terminate_process_locked()

    def diarize(self, audio_path: str | Path, settings: Settings) -> list[dict]:
        with self._lock:
            if self._unavailable_reason:
                raise DiarizationError(f"Pyannote worker unavailable: {self._unavailable_reason}")
            self._ensure_worker_locked(settings)
            proc = self._require_process()
            if proc.stdin is None:
                self._restart_worker_locked(settings)
                proc = self._require_process()
                if proc.stdin is None:
                    raise DiarizationError("Pyannote worker stdin is unavailable.")

            request = {
                "audio_path": str(Path(audio_path).resolve()),
                "num_speakers": settings.pyannote_num_speakers,
            }
            try:
                proc.stdin.write(json.dumps(request) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self._restart_worker_locked(settings)
                raise DiarizationError(f"Pyannote worker pipe failed: {exc}") from exc

            try:
                response = self._responses.get(timeout=settings.pyannote_timeout_seconds)
            except queue.Empty as exc:
                self._restart_worker_locked(settings)
                raise DiarizationError(
                    f"Pyannote diarization timed out after {settings.pyannote_timeout_seconds}s."
                ) from exc

            if response.get("ok") is True:
                segments = response.get("segments", [])
                if isinstance(segments, list):
                    return segments
                raise DiarizationError("Pyannote worker returned malformed segments.")

            message = str(response.get("error") or "Pyannote worker failed.")
            diagnostics = self._format_diagnostics()
            if diagnostics:
                message = f"{message} Last worker stderr: {diagnostics}"
            raise DiarizationError(message)

    def terminate(self) -> None:
        with self._lock:
            self._terminate_process_locked()

    def _ensure_worker_locked(self, settings: Settings) -> None:
        if self._proc is not None and self._proc.poll() is None and self._ready:
            return
        self._restart_worker_locked(settings)

    def _restart_worker_locked(self, settings: Settings) -> None:
        self._terminate_process_locked()
        self._start_worker_locked(settings)

    def _start_worker_locked(self, settings: Settings) -> None:
        self._responses = queue.Queue()
        self._stderr_lines = deque(maxlen=50)
        self._ready = False

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["MPLBACKEND"] = "Agg"
        env["HF_HUB_ETAG_TIMEOUT"] = str(settings.pyannote_hf_timeout_seconds)
        env["HF_HUB_DOWNLOAD_TIMEOUT"] = str(settings.pyannote_hf_timeout_seconds)
        env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["OMP_NUM_THREADS"] = str(settings.pyannote_worker_threads)
        env["MKL_NUM_THREADS"] = str(settings.pyannote_worker_threads)
        env["VECLIB_MAXIMUM_THREADS"] = str(settings.pyannote_worker_threads)
        env["NUMEXPR_NUM_THREADS"] = str(settings.pyannote_worker_threads)
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["VCA_PYANNOTE_STUB_TORCH_DYNAMO"] = "1" if settings.pyannote_stub_torch_dynamo else "0"
        if settings.hf_token:
            env["HF_TOKEN"] = settings.hf_token
        worker_cache_dir = settings.temp_dir / "cache" / "pyannote_worker"
        matplotlib_cache_dir = worker_cache_dir / "matplotlib"
        xdg_cache_dir = worker_cache_dir / "xdg"
        fontconfig_cache_dir = xdg_cache_dir / "fontconfig"
        for cache_dir in (matplotlib_cache_dir, xdg_cache_dir, fontconfig_cache_dir):
            cache_dir.mkdir(parents=True, exist_ok=True)
        env["MPLCONFIGDIR"] = str(matplotlib_cache_dir)
        env["XDG_CACHE_HOME"] = str(xdg_cache_dir)
        env["VCA_PYANNOTE_WORKER_CACHE_DIR"] = str(worker_cache_dir)

        worker_path = settings.backend_dir / "pyannote_worker.py"
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-u", str(worker_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=settings.root_dir,
                env=env,
            )
        except OSError as exc:
            raise DiarizationError(f"Could not start Pyannote worker: {exc}") from exc

        stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(self._proc, self._responses),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(self._proc,),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        self._wait_until_ready_locked(settings.pyannote_worker_startup_timeout_seconds)

    def _wait_until_ready_locked(self, timeout_seconds: int) -> None:
        while True:
            proc = self._require_process()
            if proc.poll() is not None:
                diagnostics = self._format_diagnostics()
                raise DiarizationError(
                    f"Pyannote worker exited during startup with code {proc.returncode}. {diagnostics}".strip()
                )
            try:
                response = self._responses.get(timeout=0.25)
            except queue.Empty:
                timeout_seconds -= 0.25
                if timeout_seconds <= 0:
                    diagnostics = self._format_diagnostics()
                    self._terminate_process_locked()
                    message = "Pyannote worker startup timed out."
                    if diagnostics:
                        message = f"{message} Last worker stderr: {diagnostics}"
                    raise DiarizationError(message)
                continue
            if response.get("ready") is True:
                self._ready = True
                self._unavailable_reason = None
                return
            if response.get("ok") is False:
                raise DiarizationError(str(response.get("error") or "Pyannote worker startup failed."))
            self._stderr_lines.append(f"unexpected startup response: {response}")

    def _read_stdout(self, proc: subprocess.Popen[str], responses: queue.Queue[dict]) -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                responses.put(json.loads(line))
            except json.JSONDecodeError:
                self._stderr_lines.append(f"invalid stdout: {line[:500]}")

    def _read_stderr(self, proc: subprocess.Popen[str]) -> None:
        if proc.stderr is None:
            return
        for line in proc.stderr:
            line = line.strip()
            if line:
                self._stderr_lines.append(line[:500])
                if line.startswith("[pyannote-worker]"):
                    print(line[:500], file=sys.stderr, flush=True)

    def _require_process(self) -> subprocess.Popen[str]:
        if self._proc is None:
            raise DiarizationError("Pyannote worker is not running.")
        return self._proc

    def _terminate_process_locked(self) -> None:
        proc = self._proc
        self._proc = None
        self._ready = False
        if proc is None:
            return
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    def _format_diagnostics(self) -> str:
        return " | ".join(list(self._stderr_lines)[-5:])


_PYANNOTE_WORKER_MANAGER = PyannoteWorkerManager()
atexit.register(_PYANNOTE_WORKER_MANAGER.terminate)


def get_pyannote_worker_manager() -> PyannoteWorkerManager:
    return _PYANNOTE_WORKER_MANAGER


def diarize(audio_path: str | Path, settings: Settings) -> list[dict]:
    return _PYANNOTE_WORKER_MANAGER.diarize(audio_path, settings)


def diarize_with_fallback(
    audio_path: str | Path,
    whisper_segments: list[dict],
    settings: Settings,
) -> tuple[list[dict], str | None]:
    try:
        return diarize(audio_path, settings), None
    except Exception as exc:  # noqa: BLE001 - fallback should catch setup/runtime failures.
        return fallback_diarization_from_whisper(whisper_segments), str(exc)
