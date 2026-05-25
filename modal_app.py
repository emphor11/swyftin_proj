from __future__ import annotations

import os

import modal


APP_NAME = "voice-call-analysis"
SECRET_NAME = "voice-call-analysis-secrets"
CACHE_VOLUME_NAME = "voice-call-analysis-cache"
OUTPUT_VOLUME_NAME = "voice-call-analysis-output"


app = modal.App(APP_NAME)

cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "build-essential",
        "cmake",
        "ffmpeg",
        "git",
        "libsndfile1",
    )
    .workdir("/app")
    .add_local_file("backend/requirements.txt", "/app/backend/requirements.txt", copy=True)
    .run_commands(
        'python -m pip install --upgrade "pip<25" "setuptools<81" wheel',
        'printf "setuptools<81\\n" > /tmp/build-constraints.txt',
        "PIP_CONSTRAINT=/tmp/build-constraints.txt python -m pip install -r /app/backend/requirements.txt",
        "mkdir -p /app/output /app/uploads /app/.tmp /app/backend/models",
    )
    .add_local_dir("backend", "/app/backend")
    .add_local_file("main.py", "/app/main.py")
    .add_local_dir("frontend/dist", "/app/frontend/dist")
)


@app.function(
    image=image,
    gpu=os.getenv("VCA_MODAL_GPU", "T4"),
    timeout=int(os.getenv("VCA_MODAL_TIMEOUT_SECONDS", "900")),
    startup_timeout=int(os.getenv("VCA_MODAL_STARTUP_TIMEOUT_SECONDS", "600")),
    secrets=[modal.Secret.from_name(SECRET_NAME)],
    volumes={
        "/root/.cache": cache_volume,
        "/app/output": output_volume,
    },
    env={
        "PYDANTIC_DISABLE_PLUGINS": "1",
        "VCA_ANALYZER_MODE": "auto",
        "VCA_LLM_RUNTIME": "hf_inference",
        "VCA_HF_INFERENCE_MODEL": "microsoft/Phi-3-mini-4k-instruct",
        "VCA_HF_INFERENCE_TIMEOUT_SECONDS": "90",
        "VCA_LLAMA_TIMEOUT_SECONDS": "120",
        "VCA_WHISPER_MODEL": "small",
        "VCA_WHISPER_LANGUAGE": "en",
        "VCA_PYANNOTE_NUM_SPEAKERS": "2",
        "VCA_PYANNOTE_TIMEOUT_SECONDS": "240",
        "VCA_PYANNOTE_WORKER_STARTUP_TIMEOUT_SECONDS": "300",
        "VCA_PYANNOTE_WARMUP_ON_STARTUP": "false",
        "VCA_PYANNOTE_DEVICE": "cuda",
        "VCA_PYANNOTE_CUDA_VISIBLE_DEVICES": "0",
        "VCA_MAX_AUDIO_SECONDS": "120",
        "VCA_CORS_ORIGIN": "*",
    },
)
@modal.asgi_app()
def web_app():
    from backend.main import app as fastapi_app

    return fastapi_app
