# SLM Choice Justification: Phi-3 Mini

## Project Requirement

This project needs a local Small Language Model that can read a customer-support call transcript and produce a structured coaching report. The model must be strong enough to follow instructions, return valid JSON, summarize the call, score the agent, identify strengths and improvement areas, and highlight key moments. It also needs to run locally because the assignment is focused on open-source/local AI rather than hosted API calls.

The selected SLM is **Phi-3 Mini 4K Instruct**. In this project, Phi-3 Mini is used through two local runtimes:

- **Default runtime:** `mlx-lm` with `mlx-community/Phi-3-mini-4k-instruct-4bit` on Apple Silicon.
- **Fallback runtime:** `llama-cpp-python` with the GGUF model `Phi-3-mini-4k-instruct-q4.gguf`.

The model choice stays the same in both cases: Phi-3 Mini. The runtime changes depending on the machine. On Apple Silicon, MLX is preferred because it uses the local Metal GPU stack and is much faster than CPU-only llama-cpp inference. The llama-cpp/GGUF path is retained because it is portable and easy to run on machines where MLX is not available.

## Architecture Diagram

```text
User uploads audio
        |
        v
FastAPI backend saves uploaded file
        |
        v
ffmpeg normalizes audio to 16 kHz mono WAV
        |
        v
Whisper transcribes speech with timestamps
        |
        v
Pyannote diarizes speakers, or fallback alternation is used
        |
        v
Transcript merger aligns Whisper text with speaker labels
        |
        v
Phi-3 Mini analyzes transcript and returns structured JSON
        |
        v
Report generator writes report.json, report.md, transcript.txt
        |
        v
React dashboard displays score, transcript, insights, and exports
```

## Why Phi-3 Mini

Phi-3 Mini is a good fit because it balances quality, size, and local runtime practicality. The analysis step is not just a simple classification task. The model has to follow a detailed schema, reason about call quality, identify sentiment, infer coaching feedback, and produce structured fields that the frontend can render. Very small models often produce generic feedback or malformed JSON. Larger models can do the task better, but they are harder to run locally within a demo-friendly time budget.

Phi-3 Mini has about 3.8B parameters, which makes it small enough for local inference while still being instruction-tuned enough for report generation. It performs well on structured prompts compared with many models in the same size class. It also has strong ecosystem support: there are GGUF quantizations for llama-cpp, and MLX-compatible quantized versions for Apple Silicon. That matters in this project because the same high-level pipeline can run in local Phi-3 mode without relying on a hosted API.

The project uses a compact single-call prompt for Phi-3. Earlier versions used three smaller LLM calls, but CPU inference was too slow. The final design uses one focused prompt, limits transcript length with `VCA_MAX_TRANSCRIPT_CHARS`, caps output with `VCA_LLAMA_MAX_TOKENS`, and runs the LLM in an isolated process with `VCA_LLAMA_TIMEOUT_SECONDS`. This keeps the user experience safe: strict Phi-3 mode fails clearly if local inference cannot complete, while Auto mode can fall back to the heuristic analyzer.

## Alternatives Considered

| Model | Strengths | Limitations for this project |
| --- | --- | --- |
| **Phi-3 Mini 4K Instruct** | Strong instruction following for its size, good structured-output behavior, local MLX and GGUF support, practical on Apple Silicon | Still slower than heuristic analysis and needs careful timeout handling |
| **Qwen2.5 3B Instruct** | Good compact model, strong multilingual ability, solid reasoning for size | Slightly less aligned with the original Phi-3/GGUF plan and required more retesting of prompts |
| **Gemma 2B** | Smaller and potentially faster | Weaker for detailed coaching feedback and more likely to produce generic or incomplete analysis |
| **Mistral 7B Instruct** | Better reasoning and richer feedback | Too heavy for the target one-day local demo, especially without a dedicated GPU |

## Runtime Decision

The original plan used `llama-cpp-python` with a quantized GGUF file. That is still supported, but on this Mac the CPU path was too slow for a practical web demo. The project therefore uses `mlx-lm` as the default runtime on Apple Silicon. This does not change the SLM choice and does not use an external API. It still runs the Phi-3 Mini model locally, but uses Apple's Metal acceleration through MLX.

The fallback GGUF path is useful for portability. If someone wants to run the project outside Apple Silicon, they can place the GGUF model in `backend/models/` and set `VCA_LLM_RUNTIME=llama_cpp`. If they are on Apple Silicon and want llama-cpp speed, they can rebuild llama-cpp-python with Metal and set `VCA_LLAMA_GPU_LAYERS=-1` after verifying that Metal offload is actually working.

## Final Decision

Phi-3 Mini was chosen because it is the best fit for a local, open-source coaching-analysis pipeline: small enough to run on a laptop, capable enough to generate detailed structured feedback, and flexible enough to run through both MLX and llama-cpp. The model is not the fastest possible option, but it gives a stronger report than a rule-based analyzer while staying realistic for the assignment's local-inference requirement.

The project also includes a heuristic analysis mode as a safety net. This is not a replacement for Phi-3; it exists so the pipeline can still complete if model loading, JSON generation, or local inference speed becomes a problem during a demo. The main quality path remains Phi-3 Mini.
