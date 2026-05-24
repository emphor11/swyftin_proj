# AI-Powered Voice Call Analysis

End-to-end voice call analysis pipeline that accepts an audio recording, normalizes and transcribes it with Whisper, labels speakers with Pyannote when available, analyzes the transcript with local Phi-3 Mini or a fast heuristic mode, and presents the result in a React dashboard.

## Current Status

This project has a working CLI, FastAPI backend, SSE progress stream, report generation, and Vite/React dashboard. The default Phi-3 path uses `mlx-lm` on Apple Silicon for local Metal-backed inference. `llama-cpp-python` remains available as a GGUF fallback runtime. Phi-3 mode is strict: if the local model cannot run, the request fails clearly instead of silently pretending a heuristic report came from Phi-3. Auto mode still falls back to the heuristic analyzer so demos never hang. Pyannote requires a Hugging Face token and accepted model terms; if it is unavailable, the pipeline falls back to speaker alternation.

Docker support is still a planned deliverable. The SLM choice writeup is available at `docs/justification.md`.

## Architecture

```text
Audio upload
  -> ffmpeg normalization
  -> Whisper transcription
  -> Pyannote diarization, or fallback speaker alternation
  -> transcript merge
  -> local Phi-3 Mini analysis, or heuristic analysis in Fast/Auto fallback mode
  -> report.json, report.md, transcript.txt
  -> React dashboard
```

## Requirements

- Python 3.10
- Node.js 18+
- ffmpeg
- Hugging Face token for Pyannote diarization
- Apple Silicon for the default MLX Phi-3 runtime, or a Phi-3 Mini GGUF model file for `llama_cpp` mode

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

On Apple Silicon, the default Phi-3 runtime is MLX:

```bash
VCA_LLM_RUNTIME=mlx .venv/bin/python main.py --transcript-file sample_audio/sample_transcript.txt --output output/phi3_mlx_smoke --analyzer-mode llm
```

The first MLX run downloads the model from Hugging Face unless it is already cached. If MLX reports `No Metal device available`, start the backend from a normal macOS Terminal session rather than a headless or sandboxed launcher.

For the first run, pre-download/cache the MLX model outside the web request so the browser does not hit the 90-second Phi-3 timeout while a ~2 GB model is still downloading:

```bash
VCA_LLM_RUNTIME=mlx VCA_LLAMA_TIMEOUT_SECONDS=1200 .venv/bin/python main.py --transcript-file sample_audio/sample_transcript.txt --output output/phi3_mlx_prewarm --analyzer-mode llm
```

After this completes once, restart the backend normally and use Phi-3 mode from the dashboard.

`llama-cpp-python` is still supported through `VCA_LLM_RUNTIME=llama_cpp`. On Apple Silicon, rebuild it with Metal before expecting usable speed:

```bash
SDKROOT="$(xcrun --show-sdk-path)"
CFLAGS="-I${SDKROOT}/usr/include/c++/v1" \
CXXFLAGS="-I${SDKROOT}/usr/include/c++/v1" \
CMAKE_ARGS="-DLLAMA_METAL=on" \
FORCE_CMAKE=1 \
.venv/bin/pip install llama-cpp-python==0.2.77 --force-reinstall --no-cache-dir --no-deps
```

Then verify the load logs mention Metal/GPU initialization and layer offload:

```bash
.venv/bin/python - <<'PY'
from llama_cpp import Llama
Llama(
    model_path="backend/models/Phi-3-mini-4k-instruct-q4.gguf",
    n_ctx=3072,
    n_threads=4,
    n_gpu_layers=-1,
    verbose=True,
)
PY
```

Only enable Metal in the app after that verification succeeds:

```bash
VCA_LLAMA_GPU_LAYERS=-1 .venv/bin/python main.py --transcript-file sample_audio/sample_transcript.txt --output output/phi3_metal_smoke --analyzer-mode llm
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

This GGUF file is only required when `VCA_LLM_RUNTIME=llama_cpp`. The default `mlx` runtime uses `VCA_MLX_MODEL_PATH`.

Create a local `.env` file in the project root:

```bash
HF_TOKEN=hf_your_token_here
VCA_ANALYZER_MODE=auto
VCA_LLM_RUNTIME=mlx
VCA_MLX_MODEL_PATH=mlx-community/Phi-3-mini-4k-instruct-4bit
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
- `llm`: strict local Phi-3 mode. It uses `VCA_LLM_RUNTIME` and fails clearly if Phi-3 cannot run or exceeds the timeout.
- `auto`: tries local Phi-3 and falls back to heuristic analysis if the model/runtime is unavailable or too slow.

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
| `VCA_LLM_RUNTIME` | `mlx` | Local Phi-3 runtime: `mlx` for Apple Silicon MLX, or `llama_cpp` for the GGUF path. |
| `VCA_MLX_MODEL_PATH` | `mlx-community/Phi-3-mini-4k-instruct-4bit` | Hugging Face model id or local path for MLX Phi-3. |
| `VCA_WHISPER_MODEL` | `small` | Whisper model name. |
| `VCA_WHISPER_LANGUAGE` | `en` | Whisper language hint. |
| `VCA_MAX_TRANSCRIPT_CHARS` | `2500` | Maximum timestamped transcript characters sent to Phi-3. |
| `VCA_LLAMA_THREADS` | `4` | CPU thread count passed to llama-cpp-python. |
| `VCA_LLAMA_CTX` | `3072` | Llama context window for the Phi-3 prompt and JSON output. |
| `VCA_LLAMA_GPU_LAYERS` | `0` | Number of layers to offload. `0` is stable CPU mode; set `-1` only after Metal verification succeeds. |
| `VCA_LLAMA_MAX_TOKENS` | `900` | Maximum JSON output tokens for Phi-3 single-call analysis. |
| `VCA_LLAMA_TIMEOUT_SECONDS` | `90` | Maximum Phi-3 analysis time before the isolated worker is terminated. `llm` mode fails clearly; `auto` mode falls back. |
| `VCA_PYANNOTE_TIMEOUT_SECONDS` | `150` | Maximum time spent trying Pyannote before falling back to speaker alternation. |
| `VCA_PYANNOTE_NUM_SPEAKERS` | `2` | Forces Pyannote to cluster at exactly N speakers. Set to `0` to let Pyannote estimate speaker count, which is slower but useful for multi-party calls. |
| `VCA_PYANNOTE_WORKER_STARTUP_TIMEOUT_SECONDS` | `300` | Maximum time for the persistent Pyannote worker to load imports/models and report ready. |
| `VCA_PYANNOTE_WORKER_WARM_FORWARD` | `false` | Optional `true`/`false` flag. When true, the worker runs a tiny silent-audio forward pass before reporting ready; disabled by default because it can make startup too slow on CPU. |
| `VCA_PYANNOTE_HF_TIMEOUT_SECONDS` | `120` | Hugging Face hub network timeout used by the worker. Increase this for manual preload runs if the model is not cached yet. |
| `VCA_PYANNOTE_WORKER_THREADS` | `4` | Thread cap for Pyannote worker numerical libraries (`OMP`, `MKL`, `vecLib`, and `numexpr`). |
| `VCA_PYANNOTE_STUB_TORCH_DYNAMO` | `true` | Installs a small worker-only `torch._dynamo` compatibility stub to avoid importing PyTorch's slow compile stack during Pyannote inference. Set to `false` if you need real `torch.compile` support in the worker. |
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

Before a demo, preload the gated Pyannote models once so startup does not hide download/auth problems:

```bash
.venv/bin/python -m backend.preload_pyannote --timeout 600
```

If this command hangs or fails, the issue is model access/cache rather than the app pipeline. Confirm that the Hugging Face account behind `HF_TOKEN` accepted the terms for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`.

If the preload hangs during import, profile the import stack:

```bash
PYTHONPROFILEIMPORTTIME=1 .venv/bin/python -c "import pyannote.audio" 2> /tmp/importtime.txt
sort -k2 -n /tmp/importtime.txt | tail -40
```

By default, diarization assumes a two-speaker support call through `VCA_PYANNOTE_NUM_SPEAKERS=2`. This avoids Pyannote's slower speaker-count estimation pass. The FastAPI backend starts a persistent Pyannote worker in the background at startup so the API port can come up immediately while Pyannote warms. The worker reports ready after the Pyannote pipeline is loaded; optional silent-audio warm inference can be enabled with `VCA_PYANNOTE_WORKER_WARM_FORWARD=true`, but it is disabled by default because it may exceed startup timeouts on CPU. CLI runs still work, but each CLI process may pay the cold-start cost because the process exits after one run.

On macOS, `pyannote.audio` can spend a long time importing `pytorch_lightning`, `lightning_fabric`, `torch._dynamo`, and `sympy` before the model load even begins. This project does not use `torch.compile`, so the worker defaults `VCA_PYANNOTE_STUB_TORCH_DYNAMO=true` to skip that compile-only import path during Pyannote inference. The worker logs these import phases to stderr so you can distinguish "slow import" from "model download/auth failure".

Very long clips, first-time model downloads, or CPU-only execution can still exceed the timeout; raise `VCA_PYANNOTE_TIMEOUT_SECONDS` if needed. If a hard crash leaves a worker behind, clean it up with:

```bash
pkill -f pyannote_worker.py
```

Phi-3 mode is the local quality mode. It stays open-source and local. The preferred Apple Silicon runtime is MLX (`VCA_LLM_RUNTIME=mlx`), which uses the local Metal stack and avoids the very slow CPU-only `llama-cpp-python` path. In strict `llm` mode, the analyzer does not silently fall back to heuristics: if MLX/llama-cpp cannot load, returns invalid JSON twice, or exceeds `VCA_LLAMA_TIMEOUT_SECONDS`, the API returns a clear error. In `auto` mode, the same failure turns into a complete heuristic fallback report with a warning. Fast mode remains the safest path when you need predictable runtime.

If MLX fails with `No Metal device available`, the Python process cannot see the Mac GPU. This commonly happens from headless or sandboxed launchers. Start the backend from a normal Terminal window and retry:

```bash
VCA_LLM_RUNTIME=mlx .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

If you choose `VCA_LLM_RUNTIME=llama_cpp`, Metal can make GGUF Phi-3 faster on Apple Silicon, but only set `VCA_LLAMA_GPU_LAYERS=-1` after the verification command shows real GPU memory and offloaded layers.

If the Metal rebuild fails with compiler errors such as `fatal error: 'cstdint' file not found` or `fatal error: 'random' file not found`, the local Xcode Command Line Tools install is incomplete or mis-selected. Repair Command Line Tools first, then rerun the Metal install command. Until that is fixed, keep demos on Fast mode or set `VCA_LLAMA_GPU_LAYERS=0` to force CPU behavior with timeout fallback.

If the rebuild succeeds but the verification logs show `using device Metal () - 0 MiB free` or `unable to allocate backend metal buffer`, the Python process cannot access usable Metal memory. Run the same verification from a normal Terminal session; if it still reports `0 MiB free`, keep `VCA_LLAMA_GPU_LAYERS=0` and use Fast mode for time-boxed demos.
