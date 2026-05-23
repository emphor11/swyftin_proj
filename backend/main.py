from __future__ import annotations

import asyncio
import json
import queue
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from .config import Settings, get_settings


REPORT_FORMATS = {
    "json": ("report.json", "application/json"),
    "md": ("report.md", "text/markdown"),
    "markdown": ("report.md", "text/markdown"),
    "transcript": ("transcript.txt", "text/plain"),
}
ALLOWED_AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"}

settings = get_settings()

app = FastAPI(
    title="AI-Powered Voice Call Analysis API",
    version="0.1.0",
    description="Upload call recordings and generate structured coaching reports.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_origin],
    allow_credentials=True,
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


def _run_pipeline(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .pipeline import run_pipeline

    return run_pipeline(*args, **kwargs)


def _safe_filename(filename: str) -> str:
    basename = Path(filename).name.strip() or "call_audio"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", basename)


def _new_job_id() -> str:
    return uuid.uuid4().hex[:12]
