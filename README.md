# AI-Powered Voice Call Analysis

End-to-end voice call analysis pipeline that accepts an audio recording, normalizes and transcribes it with Whisper, labels speakers with Pyannote when available, analyzes the transcript with Phi-3 Mini or a fast heuristic fallback, and presents the result in a React dashboard.

## Current Status

This project has a working CLI, FastAPI backend, SSE progress stream, report generation, and Vite/React dashboard. The Phi-3 path works through `llama-cpp-python`, but CPU inference is slow on this machine. Pyannote requires a Hugging Face token and accepted model terms; if it is unavailable, the pipeline falls back to speaker alternation.

Docker support and the SLM justification document are planned deliverables, but they are not present yet.

## Architecture

```text
Audio upload
  -> ffmpeg normalization
  -> Whisper transcription
  -> Pyannote diarization, or fallback speaker alternation
  -> transcript merge
  -> Phi-3 Mini analysis, or heuristic fallback
  -> report.json, report.md, transcript.txt
  -> React dashboard
```

## Requirements

- Python 3.10
- Node.js 18+
- ffmpeg
- Hugging Face token for Pyannote diarization
- Phi-3 Mini GGUF model file for LLM analysis

Install ffmpeg on macOS:

```bash
brew install ffmpeg
```

## Setup

Create and activate the Python environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Install the frontend dependencies:

```bash
cd frontend
npm install
```

Download the Phi-3 Mini GGUF model and place it here:

```text
backend/models/Phi-3-mini-4k-instruct-q4.gguf
```

Recommended source:

```text
https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
```

Create a local `.env` file in the project root:

```bash
HF_TOKEN=hf_your_token_here
VCA_ANALYZER_MODE=auto
VCA_WHISPER_MODEL=small
VCA_WHISPER_LANGUAGE=en
VCA_MODEL_PATH=backend/models/Phi-3-mini-4k-instruct-q4.gguf
VCA_CORS_ORIGIN=http://localhost:5173
```

Do not commit `.env`. It is already ignored by `.gitignore`.

## Running The App

Start the backend from the project root:

```bash
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Important: do not run `uvicorn main:app` from inside `backend/`. The backend uses package-relative imports, so it must be launched as `backend.main:app` from the project root.

Start the frontend:

```bash
cd frontend
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The Vite dev server proxies `/api` requests to `http://127.0.0.1:8000`.

## CLI Usage

Run the full audio pipeline:

```bash
.venv/bin/python main.py --input sample_audio/jfk.flac --output output/local_smoke --analyzer-mode heuristic
```

Run transcript-only smoke testing:

```bash
.venv/bin/python main.py --transcript-file sample_audio/sample_transcript.txt --output output/transcript_smoke --analyzer-mode heuristic
```

Use Phi-3 explicitly:

```bash
.venv/bin/python main.py --transcript-file sample_audio/sample_transcript.txt --output output/phi3_smoke --analyzer-mode llm
```

Analyzer modes:

- `heuristic`: fastest mode, no local LLM required.
- `llm`: requires the Phi-3 GGUF model and fails if it cannot run.
- `auto`: uses Phi-3 when the model file exists, otherwise falls back to heuristic analysis.

## API

Health check:

```bash
curl http://127.0.0.1:8000/api/health
```

Upload and analyze audio:

```bash
curl -N -X POST \
  -F "file=@sample_audio/jfk.flac" \
  "http://127.0.0.1:8000/api/analyze?analyzer_mode=heuristic"
```

List generated reports:

```bash
curl http://127.0.0.1:8000/api/reports
```

Fetch one report:

```bash
curl http://127.0.0.1:8000/api/reports/<report_id>
```

Download report files:

```bash
curl -O http://127.0.0.1:8000/api/reports/<report_id>/download/json
curl -O http://127.0.0.1:8000/api/reports/<report_id>/download/md
curl -O http://127.0.0.1:8000/api/reports/<report_id>/download/transcript
```

## Output

Each run writes a report directory under `output/<report_id>/`:

```text
report.json
report.md
transcript.txt
<uploaded-audio-file>
```

`output/` is gitignored except for `output/.gitkeep`.

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | none | Hugging Face token for Pyannote. |
| `VCA_MODEL_PATH` | `backend/models/Phi-3-mini-4k-instruct-q4.gguf` | Phi-3 GGUF path. |
| `VCA_ANALYZER_MODE` | `auto` | Default analyzer mode: `auto`, `llm`, or `heuristic`. |
| `VCA_WHISPER_MODEL` | `small` | Whisper model name. |
| `VCA_WHISPER_LANGUAGE` | `en` | Whisper language hint. |
| `VCA_MAX_TRANSCRIPT_CHARS` | `12000` | Parsed by settings; current Phi-3 prompts truncate to 3000 characters internally. |
| `VCA_LLAMA_THREADS` | `4` | Parsed by settings; current Llama initialization uses 4 threads. |
| `VCA_LLAMA_CTX` | `4096` | Parsed by settings; current Llama initialization uses 4096 context. |
| `VCA_PYANNOTE_TIMEOUT_SECONDS` | `90` | Maximum time spent trying Pyannote before falling back to speaker alternation. |
| `VCA_PYANNOTE_NUM_SPEAKERS` | `2` | Forces Pyannote to cluster at exactly N speakers. Set to `0` to let Pyannote estimate speaker count, which is slower but useful for multi-party calls. |
| `VCA_CORS_ORIGIN` | `http://localhost:5173` | Frontend origin allowed by FastAPI CORS. |

## Troubleshooting

If the dashboard says `llama-cpp-python is not installed`, the backend is probably running from the wrong Python environment. Stop the server on port 8000 and restart it from the project root with:

```bash
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Check which process is bound to port 8000:

```bash
lsof -nP -iTCP:8000
```

If Pyannote fails, verify all three:

- `HF_TOKEN` is present in `.env`.
- You accepted the Pyannote model terms on Hugging Face.
- `pyannote.audio` is installed in the same `.venv` used to run the backend.

By default, diarization assumes a two-speaker support call through `VCA_PYANNOTE_NUM_SPEAKERS=2`. This avoids Pyannote's slower speaker-count estimation pass. Very long clips, first-time model downloads, or CPU-only execution can still exceed the timeout; raise `VCA_PYANNOTE_TIMEOUT_SECONDS` if needed.

If Phi-3 is slow, use `--analyzer-mode heuristic` for smoke tests and demos. The local Phi-3 CPU path is functional but can take many minutes for the three-call analysis strategy.
