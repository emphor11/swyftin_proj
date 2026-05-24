from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import tempfile
import types
import wave
from pathlib import Path
from typing import Any


PROTOCOL_STDOUT = sys.stdout


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _log_phase(message: str) -> None:
    print(f"[pyannote-worker] {message}", file=sys.stderr, flush=True)


def _configure_process() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    cache_root = Path(os.environ.get("VCA_PYANNOTE_WORKER_CACHE_DIR", tempfile.gettempdir()))
    matplotlib_cache_dir = cache_root / "matplotlib"
    xdg_cache_dir = cache_root / "xdg"
    fontconfig_cache_dir = xdg_cache_dir / "fontconfig"
    for cache_dir in (matplotlib_cache_dir, xdg_cache_dir, fontconfig_cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))
    logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
    for logger_name in (
        "pyannote",
        "pyannote.audio",
        "pytorch_lightning",
        "lightning",
        "huggingface_hub",
        "transformers",
        "torch",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


def _emit(payload: dict[str, Any]) -> None:
    PROTOCOL_STDOUT.write(json.dumps(payload) + "\n")
    PROTOCOL_STDOUT.flush()


def _install_torch_dynamo_stub(torch_module) -> None:
    if not _env_flag("VCA_PYANNOTE_STUB_TORCH_DYNAMO", default=True):
        return
    if "torch._dynamo" in sys.modules:
        return

    dynamo_stub = types.ModuleType("torch._dynamo")

    class OptimizedModule(torch_module.nn.Module):
        pass

    def identity_decorator(fn=None, *args, **kwargs):
        if fn is not None:
            return fn
        return lambda wrapped: wrapped

    dynamo_stub.OptimizedModule = OptimizedModule
    dynamo_stub.disable = identity_decorator
    dynamo_stub.graph_break = lambda *args, **kwargs: None
    sys.modules["torch._dynamo"] = dynamo_stub
    setattr(torch_module, "_dynamo", dynamo_stub)
    _log_phase("torch._dynamo stub installed")


def _create_silent_wav() -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="pyannote_warmup_", suffix=".wav", delete=False)
    path = Path(handle.name)
    handle.close()

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)

    return path


def _load_pipeline():
    _log_phase("loading pyannote pipeline")
    with contextlib.redirect_stdout(sys.stderr):
        try:
            _log_phase("importing numpy")
            import numpy as np

            if not hasattr(np, "NAN"):
                np.NAN = np.nan
            _log_phase("numpy imported")
        except ImportError:
            _log_phase("numpy import skipped")

        _log_phase("importing torch")
        import torch

        _log_phase("torch imported")
        _install_torch_dynamo_stub(torch)
        _log_phase("importing scipy")
        import scipy  # noqa: F401

        _log_phase("scipy imported")
        _log_phase("importing matplotlib")
        import matplotlib  # noqa: F401

        _log_phase("matplotlib imported")
        _log_phase("importing huggingface_hub")
        import huggingface_hub  # noqa: F401

        _log_phase("huggingface_hub imported")
        _log_phase("importing lightning_fabric")
        import lightning_fabric  # noqa: F401

        _log_phase("lightning_fabric imported")
        _log_phase("importing pytorch_lightning")
        import pytorch_lightning  # noqa: F401

        _log_phase("pytorch_lightning imported")
        _log_phase("importing pyannote.core")
        import pyannote.core  # noqa: F401

        _log_phase("pyannote.core imported")

        _log_phase("importing pyannote.audio")
        from pyannote.audio import Pipeline
        _log_phase("pyannote.audio imported")

        _log_phase("calling Pipeline.from_pretrained")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.environ.get("HF_TOKEN") or None,
        )
    if pipeline is None:
        raise RuntimeError("Could not load pyannote/speaker-diarization-3.1.")
    _log_phase("pyannote pipeline loaded")
    return pipeline


def _warm_forward_pass(pipeline) -> None:
    _log_phase("warm forward pass started")
    path = _create_silent_wav()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            pipeline(str(path), num_speakers=1)
    except Exception as exc:  # noqa: BLE001 - warmup silence can fail without breaking real audio.
        print(f"Pyannote warmup forward pass failed: {exc}", file=sys.stderr, flush=True)
    finally:
        try:
            path.unlink()
        except OSError:
            pass
    _log_phase("warm forward pass finished")


def _run_job(pipeline, request: dict[str, Any]) -> dict[str, Any]:
    audio_path = Path(str(request.get("audio_path", "")))
    if not audio_path.is_absolute():
        return {"ok": False, "error": "audio_path must be absolute"}
    if not audio_path.exists():
        return {"ok": False, "error": f"audio file not found: {audio_path}"}

    try:
        num_speakers = int(request.get("num_speakers", 2))
    except (TypeError, ValueError):
        return {"ok": False, "error": "num_speakers must be an integer"}

    kwargs: dict[str, int] = {}
    if num_speakers > 0:
        kwargs["num_speakers"] = num_speakers

    segments = []
    with contextlib.redirect_stdout(sys.stderr):
        diarization = pipeline(str(audio_path), **kwargs)
        tracks = list(diarization.itertracks(yield_label=True))

    for turn, _, speaker in tracks:
        segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )

    return {"ok": True, "segments": segments}


def main() -> int:
    _configure_process()
    pipeline = _load_pipeline()
    if _env_flag("VCA_PYANNOTE_WORKER_WARM_FORWARD"):
        _warm_forward_pass(pipeline)
    _emit({"ready": True})
    _log_phase("ready emitted")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = _run_job(pipeline, request)
        except Exception as exc:  # noqa: BLE001 - protocol should report worker errors as JSON.
            response = {"ok": False, "error": str(exc)}
        _emit(response)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
