from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import re
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "1")

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from .config import Settings, get_settings
from .transcriber import get_audio_duration, get_pyannote_worker_manager


REPORT_FORMATS = {
    "json": ("report.json", "application/json"),
    "md": ("report.md", "text/markdown"),
    "markdown": ("report.md", "text/markdown"),
    "transcript": ("transcript.txt", "text/plain"),
}
ALLOWED_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}

settings = get_settings()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    manager = get_pyannote_worker_manager()

    if settings.pyannote_warmup_on_startup:
        def warmup_pyannote() -> None:
            try:
                manager.warmup(settings)
            except Exception as exc:  # noqa: BLE001 - backend should still run with diarization fallback.
                logger.warning("Pyannote worker warmup failed; fallback diarization remains available: %s", exc)
                manager.mark_unavailable(str(exc))

        warmup_thread = threading.Thread(
            target=warmup_pyannote,
            name="pyannote-worker-warmup",
            daemon=True,
        )
        warmup_thread.start()

    try:
        yield
    finally:
        await asyncio.to_thread(manager.terminate)

app = FastAPI(
    title="AI-Powered Voice Call Analysis API",
    version="0.1.0",
    description="Upload call recordings and generate structured coaching reports.",
    lifespan=lifespan,
)

cors_origins = [origin.strip() for origin in settings.cors_origin.split(",") if origin.strip()]
if not cors_origins:
    cors_origins = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze_call(
    file: UploadFile = File(...),
    analyzer_mode: str | None = Query(default=None, pattern="^(auto|llm|heuristic)$"),
) -> EventSourceResponse:
    job_id = _new_job_id()
    filename = _safe_filename(file.filename or "call_audio")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_AUDIO_SUFFIXES))
        raise HTTPException(status_code=400, detail=f"Unsupported audio format. Allowed: {allowed}")

    upload_path = settings.upload_dir / job_id / filename
    await _save_upload(file, upload_path)
    _validate_audio_duration(upload_path)

    report_dir = settings.output_dir / job_id
    return EventSourceResponse(
        _pipeline_event_stream(
            input_path=upload_path,
            report_dir=report_dir,
            analyzer_mode=analyzer_mode,
            settings=settings,
        )
    )


@app.get("/api/reports")
def list_reports() -> dict[str, list[dict[str, Any]]]:
    _reload_output_volume()

    reports = []
    report_dirs = [path for path in settings.output_dir.iterdir() if path.is_dir()]
    for report_dir in sorted(report_dirs, key=lambda path: path.stat().st_mtime, reverse=True):
        if not report_dir.is_dir():
            continue
        report_json = report_dir / "report.json"
        if not report_json.exists():
            continue

        analysis = _read_json(report_json)
        created_at = datetime.fromtimestamp(report_json.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
        reports.append(
            {
                "id": report_dir.name,
                "created_at": created_at,
                "agent_score": analysis.get("agent_score"),
                "overall_sentiment": analysis.get("overall_sentiment"),
                "analysis_mode": analysis.get("analysis_mode"),
                "has_markdown": (report_dir / "report.md").exists(),
                "has_transcript": (report_dir / "transcript.txt").exists(),
            }
        )

    return {"reports": reports}


@app.get("/api/reports/{report_id}")
def get_report(report_id: str) -> dict[str, Any]:
    _reload_output_volume()

    report_dir = _resolve_report_dir(report_id)
    report_json = report_dir / "report.json"
    if not report_json.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    return {
        "id": report_id,
        "analysis": _read_json(report_json),
        "markdown": _read_optional_text(report_dir / "report.md"),
        "transcript": _read_optional_text(report_dir / "transcript.txt"),
        "files": {
            "json": f"/api/reports/{report_id}/download/json",
            "markdown": f"/api/reports/{report_id}/download/md",
            "transcript": f"/api/reports/{report_id}/download/transcript",
        },
    }


@app.get("/api/reports/{report_id}/download/{report_format}")
def download_report(report_id: str, report_format: str) -> FileResponse:
    if report_format not in REPORT_FORMATS:
        raise HTTPException(status_code=400, detail="Format must be one of: json, md, markdown, transcript")

    filename, media_type = REPORT_FORMATS[report_format]
    path = _resolve_report_dir(report_id) / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report file not found")

    return FileResponse(
        path,
        media_type=media_type,
        filename=f"{report_id}_{filename}",
    )


if settings.frontend_dist_dir.exists():
    assets_dir = settings.frontend_dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/", include_in_schema=False, response_model=None)
    def serve_frontend_root():
        return FileResponse(settings.frontend_dist_dir / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False, response_model=None)
    def serve_frontend(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")

        candidate = settings.frontend_dist_dir / full_path
        if candidate.is_file():
            return FileResponse(candidate)

        index_path = settings.frontend_dist_dir / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return HTMLResponse("Frontend build not found.", status_code=404)


async def _pipeline_event_stream(
    input_path: Path,
    report_dir: Path,
    analyzer_mode: str | None,
    settings: Settings,
) -> Any:
    events: queue.Queue[dict[str, Any] | None] = queue.Queue()

    def emit(stage: str, progress: int, message: str) -> None:
        if stage == "complete":
            return
        events.put({"stage": stage, "progress": progress, "message": message})

    def worker() -> None:
        try:
            result = _run_pipeline(
                input_path,
                output_dir=report_dir,
                analyzer_mode=analyzer_mode,
                settings=settings,
                progress_callback=emit,
            )
            _commit_output_volume()
            events.put(
                {
                    "stage": "complete",
                    "progress": 100,
                    "message": "Analysis complete.",
                    "report_id": result["id"],
                    "report_url": f"/api/reports/{result['id']}",
                    "metadata": result["metadata"],
                    "paths": result["paths"],
                }
            )
        except Exception as exc:  # pragma: no cover - exercised through integration runs.
            events.put(
                {
                    "stage": "error",
                    "progress": 100,
                    "message": str(exc),
                    "error": exc.__class__.__name__,
                }
            )
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        event = await asyncio.to_thread(events.get)
        if event is None:
            break
        event_name = "error" if event["stage"] == "error" else "progress"
        yield {"event": event_name, "data": json.dumps(event)}


async def _save_upload(file: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        while chunk := await file.read(1024 * 1024):
            handle.write(chunk)
    await file.close()


def _validate_audio_duration(path: Path) -> None:
    if settings.max_audio_seconds <= 0:
        return

    duration = get_audio_duration(path)
    if duration <= 0:
        return
    if duration <= settings.max_audio_seconds:
        return

    try:
        path.unlink()
    except OSError:
        pass
    raise HTTPException(
        status_code=413,
        detail=(
            f"Audio is {duration:.0f}s long. Hosted demo uploads are limited to "
            f"{settings.max_audio_seconds}s to keep processing timely."
        ),
    )


def _resolve_report_dir(report_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", report_id):
        raise HTTPException(status_code=404, detail="Report not found")

    report_dir = (settings.output_dir / report_id).resolve()
    output_dir = settings.output_dir.resolve()
    if report_dir != output_dir and output_dir not in report_dir.parents:
        raise HTTPException(status_code=404, detail="Report not found")
    return report_dir


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid report JSON: {path.name}") from exc


def _read_optional_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _commit_output_volume() -> None:
    _modal_output_volume_action("commit")


def _reload_output_volume() -> None:
    _modal_output_volume_action("reload")


def _modal_output_volume_action(action: str) -> None:
    volume_name = settings.modal_output_volume_name
    if not volume_name:
        return

    try:
        import modal  # type: ignore[import-not-found]

        volume = modal.Volume.from_name(volume_name)
        if action == "commit":
            volume.commit()
        elif action == "reload":
            volume.reload()
    except Exception as exc:  # noqa: BLE001 - local/Docker runs should not fail on Modal-only persistence hooks.
        logger.warning("Modal output volume %s failed for %s: %s", action, volume_name, exc)


def _run_pipeline(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .pipeline import run_pipeline

    return run_pipeline(*args, **kwargs)


def _safe_filename(filename: str) -> str:
    basename = Path(filename).name.strip() or "call_audio"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", basename)


def _new_job_id() -> str:
    return uuid.uuid4().hex[:12]
