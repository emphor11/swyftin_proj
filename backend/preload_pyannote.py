from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
import types
import wave
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency exists in normal project env.
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parents[1]


def _log(message: str) -> None:
    print(message, flush=True)


def _create_silent_wav() -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="pyannote_preload_", suffix=".wav", delete=False)
    path = Path(handle.name)
    handle.close()

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)

    return path


def _configure_env(timeout_seconds: int) -> None:
    if load_dotenv:
        load_dotenv(ROOT_DIR / ".env")

    cache_root = ROOT_DIR / ".tmp" / "cache" / "pyannote_preload"
    matplotlib_cache_dir = cache_root / "matplotlib"
    xdg_cache_dir = cache_root / "xdg"
    fontconfig_cache_dir = xdg_cache_dir / "fontconfig"
    for cache_dir in (matplotlib_cache_dir, xdg_cache_dir, fontconfig_cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("VCA_PYANNOTE_STUB_TORCH_DYNAMO", "1")
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(timeout_seconds)
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout_seconds)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    _log("torch._dynamo stub installed.")


def _patch_huggingface_hub_compat(huggingface_hub_module) -> None:
    """Allow pyannote 3.x to run with huggingface-hub 1.x."""
    original_download = huggingface_hub_module.hf_hub_download

    def compat_hf_hub_download(*args, **kwargs):
        auth_token = kwargs.pop("use_auth_token", None)
        if auth_token and "token" not in kwargs:
            kwargs["token"] = auth_token
        return original_download(*args, **kwargs)

    huggingface_hub_module.hf_hub_download = compat_hf_hub_download
    file_download = getattr(huggingface_hub_module, "file_download", None)
    if file_download is not None:
        file_download.hf_hub_download = compat_hf_hub_download
    _log("huggingface_hub use_auth_token compatibility patch installed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preload and cache Pyannote diarization models.")
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Hugging Face hub timeout in seconds while preloading.",
    )
    parser.add_argument(
        "--warm-forward",
        action="store_true",
        help="Run a tiny silent-audio diarization after loading the pipeline.",
    )
    args = parser.parse_args()

    _configure_env(args.timeout)
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is missing. Add it to .env before preloading Pyannote.", file=sys.stderr)
        return 1

    _log("Loading pyannote/speaker-diarization-3.1...")
    with contextlib.redirect_stdout(sys.stderr):
        try:
            _log("Importing numpy...")
            import numpy as np

            if not hasattr(np, "NAN"):
                np.NAN = np.nan
            _log("numpy imported.")
        except ImportError:
            _log("numpy import skipped.")

        _log("Importing torch...")
        import torch

        _log("torch imported.")
        _install_torch_dynamo_stub(torch)
        _log("Importing scipy...")
        import scipy  # noqa: F401

        _log("scipy imported.")
        _log("Importing matplotlib...")
        import matplotlib  # noqa: F401

        _log("matplotlib imported.")
        _log("Importing huggingface_hub...")
        import huggingface_hub

        _log("huggingface_hub imported.")
        _patch_huggingface_hub_compat(huggingface_hub)
        _log("Importing lightning_fabric...")
        import lightning_fabric  # noqa: F401

        _log("lightning_fabric imported.")
        _log("Importing pytorch_lightning...")
        import pytorch_lightning  # noqa: F401

        _log("pytorch_lightning imported.")
        _log("Importing pyannote.core...")
        import pyannote.core  # noqa: F401

        _log("pyannote.core imported.")
        _log("Importing pyannote.audio...")
        from pyannote.audio import Pipeline
        _log("pyannote.audio imported.")

        _log("Calling Pipeline.from_pretrained...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )

    if pipeline is None:
        print(
            "Could not load pyannote/speaker-diarization-3.1. "
            "Verify HF_TOKEN and accept gated model terms.",
            file=sys.stderr,
        )
        return 1

    _log("Pyannote pipeline loaded.")
    if args.warm_forward:
        path = _create_silent_wav()
        try:
            _log("Running silent warm-forward pass...")
            with contextlib.redirect_stdout(sys.stderr):
                pipeline(str(path), num_speakers=1)
            _log("Warm-forward pass completed.")
        finally:
            try:
                path.unlink()
            except OSError:
                pass

    _log("Pyannote preload completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
