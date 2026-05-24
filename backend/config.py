from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency exists in the runtime env.
    load_dotenv = None

if load_dotenv:
    load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT_DIR
    backend_dir: Path = BACKEND_DIR
    output_dir: Path = ROOT_DIR / "output"
    upload_dir: Path = ROOT_DIR / "uploads"
    temp_dir: Path = ROOT_DIR / ".tmp"
    model_dir: Path = BACKEND_DIR / "models"
    phi3_model_path: Path = Path(
        os.getenv(
            "VCA_MODEL_PATH",
            str(BACKEND_DIR / "models" / "Phi-3-mini-4k-instruct-q4.gguf"),
        )
    )
    whisper_model: str = os.getenv("VCA_WHISPER_MODEL", "small")
    whisper_language: str | None = os.getenv("VCA_WHISPER_LANGUAGE", "en")
    hf_token: str | None = os.getenv("HF_TOKEN")
    analyzer_mode: str = os.getenv("VCA_ANALYZER_MODE", "auto")
    max_transcript_chars: int = int(os.getenv("VCA_MAX_TRANSCRIPT_CHARS", "12000"))
    llama_threads: int = int(os.getenv("VCA_LLAMA_THREADS", "4"))
    llama_ctx: int = int(os.getenv("VCA_LLAMA_CTX", "4096"))
    pyannote_timeout_seconds: int = int(os.getenv("VCA_PYANNOTE_TIMEOUT_SECONDS", "90"))
    pyannote_num_speakers: int = int(os.getenv("VCA_PYANNOTE_NUM_SPEAKERS", "2"))
    pyannote_worker_startup_timeout_seconds: int = int(
        os.getenv("VCA_PYANNOTE_WORKER_STARTUP_TIMEOUT_SECONDS", "300")
    )
    pyannote_hf_timeout_seconds: int = int(os.getenv("VCA_PYANNOTE_HF_TIMEOUT_SECONDS", "120"))
    pyannote_worker_threads: int = int(os.getenv("VCA_PYANNOTE_WORKER_THREADS", "4"))
    pyannote_stub_torch_dynamo: bool = os.getenv("VCA_PYANNOTE_STUB_TORCH_DYNAMO", "true").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    cors_origin: str = os.getenv("VCA_CORS_ORIGIN", "http://localhost:5173")


def get_settings() -> Settings:
    settings = Settings()
    ensure_directories(settings)
    return settings


def ensure_directories(settings: Settings) -> None:
    for path in (
        settings.output_dir,
        settings.upload_dir,
        settings.temp_dir,
        settings.model_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
