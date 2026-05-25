from __future__ import annotations

import json
import multiprocessing
import queue
import re
import traceback
from functools import lru_cache
from typing import Any

from .config import Settings
from .merger import format_timestamp


ANALYSIS_PROMPT = """<|system|>
You are a senior call center quality analyst. You output ONLY one valid JSON object. Never include prose, apologies, markdown, code fences, headings, or comments. Your reply must begin with {{ and end with }}.
<|end|>
<|user|>
Below is a transcript of a two-speaker customer support call (Agent and Customer). Read it carefully, then return your analysis as one JSON object that follows the SCHEMA below.

SCHEMA - the keys, structure, and value types are fixed. The example values shown are illustrative only. Replace every value with content derived from the transcript.

{{
  "call_summary": "Two or three sentences covering what the customer needed, what the agent did, and how the call ended.",
  "overall_sentiment": "positive",
  "customer_sentiment_journey": "Short arc such as frustrated -> reassured -> satisfied.",
  "agent_score": 7,
  "score_breakdown": {{
    "greeting_and_opening": 8,
    "active_listening": 7,
    "problem_resolution": 7,
    "professionalism": 8,
    "closing": 6
  }},
  "strengths": [
    "Concrete thing the agent did well, referencing what they actually said.",
    "Another specific strength tied to a real moment in the transcript.",
    "A third specific strength if the evidence supports it."
  ],
  "improvement_areas": [
    "Specific weakness, plus what the agent should do differently next time.",
    "Another concrete improvement tied to a real moment in the transcript."
  ],
  "recommended_next_steps": [
    "One concrete coaching action the agent can apply on the next call.",
    "Another concrete coaching action.",
    "A third coaching action if useful."
  ],
  "key_moments": [
    {{
      "timestamp_range": "use a real range from the transcript such as 0:18-0:32",
      "speaker": "Agent",
      "description": "What happened in this moment and why it matters.",
      "impact": "positive"
    }}
  ],
  "compliance_flags": []
}}

RULES:
- Reply with exactly one JSON object. No text before it. No text after it.
- Use the exact key names shown above. Use double quotes around every key and string value.
- All scores are integers from 1 to 10. Do not return decimals or strings for scores.
- Do not copy the example score values from the schema. Calculate every score from transcript evidence.
- The agent_score must be consistent with the score_breakdown and should be close to the rounded average of the five category scores.
- Be fair and conservative. Use 9 or 10 only when the transcript shows clearly excellent performance.
- "overall_sentiment" must be one of: positive, negative, neutral, mixed.
- "impact" inside key_moments must be one of: positive, negative, neutral.
- Use 2 or 3 items in strengths, improvement_areas, and recommended_next_steps.
- Use 1 to 3 entries in key_moments. Copy timestamp_range from the transcript exactly.
- "compliance_flags" is [] when no issues are present, otherwise a short list of issue descriptions.
- Strengths and improvement_areas must reference specific things the agent said or did in THIS transcript. Do not give generic advice that would fit any call.
- Never output a category name alone, for example "active_listening", as a strength or improvement.

TRANSCRIPT:
{transcript}
<|end|>
<|assistant|>
"""


DEFAULT_BREAKDOWN = {
    "greeting_and_opening": 5,
    "active_listening": 5,
    "problem_resolution": 5,
    "professionalism": 5,
    "closing": 5,
}


def _parse_json(raw: str) -> dict[str, Any]:
    candidates = [raw.strip()]
    candidates.append(re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE).replace("```", "").strip())

    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        candidates.append(match.group(0).strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return {
        "error": "parse_failed",
        "raw": raw,
    }


def parse_json_response(raw: str) -> dict[str, Any]:
    return _parse_json(raw)


def _truncate_transcript(transcript: str, max_chars: int) -> str:
    if len(transcript) <= max_chars:
        return transcript

    marker = "[transcript truncated]"
    budget = max(0, max_chars - len(marker) - 2)
    head_budget = budget // 2
    tail_budget = budget - head_budget
    lines = transcript.splitlines()

    head_lines: list[str] = []
    head_size = 0
    for line in lines:
        next_size = head_size + len(line) + (1 if head_lines else 0)
        if next_size > head_budget:
            break
        head_lines.append(line)
        head_size = next_size

    tail_lines: list[str] = []
    tail_size = 0
    for line in reversed(lines):
        next_size = tail_size + len(line) + (1 if tail_lines else 0)
        if next_size > tail_budget:
            break
        tail_lines.append(line)
        tail_size = next_size

    if head_lines or tail_lines:
        return "\n".join([*head_lines, marker, *reversed(tail_lines)]).strip()

    return transcript[:max_chars].rstrip()


def _format_transcript(blocks: list[dict], max_chars: int = 2500) -> str:
    lines = []
    for block in blocks:
        speaker = str(block.get("speaker", "Speaker")).strip() or "Speaker"
        text = str(block.get("text", "")).strip()
        if text:
            start = format_timestamp(block.get("start", 0))
            end = format_timestamp(block.get("end", 0))
            lines.append(f"[{start}-{end}] {speaker}: {text}")

    transcript = "\n".join(lines)
    return _truncate_transcript(transcript, max_chars)


def normalize_analysis_schema(analysis: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(analysis)

    normalized.setdefault("call_summary", "The call was analyzed, but the summary was not returned.")
    normalized.setdefault("overall_sentiment", "neutral")
    normalized.setdefault("customer_sentiment_journey", "Not enough information to determine a journey.")
    normalized.setdefault("agent_score", 5)
    normalized.setdefault("score_breakdown", {})
    normalized["score_breakdown"] = {
        **DEFAULT_BREAKDOWN,
        **{
            key: _clamp_score(value)
            for key, value in dict(normalized.get("score_breakdown", {})).items()
        },
    }
    normalized["agent_score"] = _clamp_score(normalized.get("agent_score", 5))

    normalized.setdefault("compliance_flags", [])

    for key in ("strengths", "improvement_areas", "recommended_next_steps", "key_moments", "compliance_flags"):
        value = normalized.get(key)
        if not isinstance(value, list):
            normalized[key] = [] if value in (None, "") else [str(value)]

    return normalized


def enrich_analysis_from_transcript(analysis: dict[str, Any], blocks: list[dict]) -> dict[str, Any]:
    enriched = normalize_analysis_schema(analysis)
    agent_text = " ".join(str(block.get("text", "")) for block in blocks if block.get("speaker") == "Agent").lower()
    first_agent = next((block for block in blocks if block.get("speaker") == "Agent"), None)

    if not enriched.get("strengths"):
        enriched["strengths"] = _fallback_strengths(
            first_agent,
            enriched["score_breakdown"].get("active_listening", 5),
            enriched["score_breakdown"].get("problem_resolution", 5),
        )

    if not enriched.get("improvement_areas"):
        enriched["improvement_areas"] = _fallback_improvements(enriched["score_breakdown"])

    if not enriched.get("recommended_next_steps"):
        enriched["recommended_next_steps"] = [
            "Open with ownership language that makes the customer feel the issue is being handled.",
            "Restate the customer's issue before explaining the fix or next action.",
            "Close with a recap of the outcome, timeline, and a final check for questions.",
        ]

    if not enriched.get("key_moments"):
        enriched["key_moments"] = _fallback_key_moments(blocks)

    if not enriched.get("compliance_flags"):
        enriched["compliance_flags"] = _fallback_compliance_flags(agent_text)

    return enriched


def _clamp_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 5
    return max(1, min(10, score))


@lru_cache(maxsize=1)
def _load_llama_cpp(model_path: str, n_ctx: int, n_threads: int, n_gpu_layers: int):
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError("llama-cpp-python is not installed.") from exc

    try:
        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
    except Exception:
        if n_gpu_layers == 0:
            raise
        return Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=0,
            verbose=False,
        )


@lru_cache(maxsize=1)
def _load_mlx_model(model_path: str):
    try:
        from mlx_lm import load
    except Exception as exc:  # noqa: BLE001 - MLX fails early when Metal is not visible.
        raise RuntimeError(f"mlx-lm is not available or cannot access Metal: {exc}") from exc

    try:
        return load(model_path)
    except Exception as exc:  # noqa: BLE001 - surface HF/model/Metal failures cleanly.
        raise RuntimeError(f"Could not load MLX model '{model_path}': {exc}") from exc


def _run_llama_cpp_json(llm: Any, prompt: str, max_tokens: int, temperature: float) -> dict[str, Any]:
    response = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|end|>"],
    )
    raw_output = response["choices"][0]["text"].strip()
    parsed = _parse_json(raw_output)
    if not isinstance(parsed, dict):
        return {
            "error": "parse_failed",
            "raw": raw_output,
        }
    return parsed


def _run_mlx_json(model: Any, tokenizer: Any, prompt: str, max_tokens: int, temperature: float) -> dict[str, Any]:
    try:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
    except Exception as exc:  # noqa: BLE001 - MLX import can fail when Metal is unavailable.
        raise RuntimeError(f"mlx-lm generation is not available: {exc}") from exc

    sampler = make_sampler(temp=temperature, top_p=0.9 if temperature > 0 else 0.0)
    raw_output = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    )

    raw_text = str(raw_output).strip()
    parsed = _parse_json(raw_text)
    if not isinstance(parsed, dict):
        return {
            "error": "parse_failed",
            "raw": raw_text,
        }
    return parsed


def _prompt_to_chat_messages(prompt: str) -> list[dict[str, str]]:
    system_match = re.search(r"<\|system\|>\s*([\s\S]*?)\s*<\|end\|>", prompt)
    user_match = re.search(r"<\|user\|>\s*([\s\S]*?)\s*<\|end\|>", prompt)
    if system_match and user_match:
        return [
            {"role": "system", "content": system_match.group(1).strip()},
            {"role": "user", "content": user_match.group(1).strip()},
        ]
    return [{"role": "user", "content": prompt}]


def _extract_chat_content(response: Any) -> str:
    choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
    if not choices:
        return str(response).strip()

    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
    if message is None:
        text = choice.get("text") if isinstance(choice, dict) else getattr(choice, "text", "")
        return str(text).strip()

    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
    return str(content).strip()


def _run_hf_inference_json(payload: dict[str, Any], prompt: str, max_tokens: int, temperature: float) -> dict[str, Any]:
    try:
        from huggingface_hub import InferenceClient
    except Exception as exc:  # noqa: BLE001 - dependency/token/provider failures are surfaced clearly.
        raise RuntimeError(f"huggingface_hub inference client is not available: {exc}") from exc

    token = str(payload.get("hf_token") or "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required for VCA_LLM_RUNTIME=hf_inference.")

    model_id = str(payload["hf_inference_model"])
    provider = payload.get("hf_inference_provider") or None
    timeout = float(payload["hf_inference_timeout_seconds"])
    client = InferenceClient(
        model=model_id,
        provider=provider,
        token=token,
        timeout=timeout,
    )
    messages = _prompt_to_chat_messages(prompt)

    request = {
        "messages": messages,
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9 if temperature > 0 else 0.1,
        "stop": ["<|end|>"],
    }
    try:
        response = client.chat_completion(
            **request,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001 - some providers reject response_format.
        if "response_format" not in str(exc).lower():
            raise
        response = client.chat_completion(**request)

    raw_text = _extract_chat_content(response)
    parsed = _parse_json(raw_text)
    if not isinstance(parsed, dict):
        return {
            "error": "parse_failed",
            "raw": raw_text,
        }
    return parsed


@lru_cache(maxsize=1)
def _load_transformers_model(model_id: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001 - deployment images may omit the hosted runtime deps.
        raise RuntimeError(f"Transformers Phi-3 runtime is not available: {exc}") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("VCA_LLM_RUNTIME=transformers_cuda requires a CUDA GPU.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    except Exception as exc:  # noqa: BLE001 - surface HF/model load failures cleanly.
        raise RuntimeError(f"Could not load Transformers Phi-3 model '{model_id}': {exc}") from exc

    model.eval()
    return model, tokenizer


def _run_transformers_json(model: Any, tokenizer: Any, prompt: str, max_tokens: int, temperature: float) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - torch import failures should be explicit.
        raise RuntimeError(f"Torch is not available for Transformers generation: {exc}") from exc

    try:
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]

        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": 0.9,
                }
            )

        with torch.inference_mode():
            output_ids = model.generate(**generation_kwargs)

        generated_ids = output_ids[0][input_length:]
        raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    except Exception as exc:  # noqa: BLE001 - generation failures should reach the UI clearly.
        raise RuntimeError(f"Transformers Phi-3 generation failed: {exc}") from exc

    raw_text = raw_text.split("<|end|>", 1)[0].strip()
    parsed = _parse_json(raw_text)
    if not isinstance(parsed, dict):
        return {
            "error": "parse_failed",
            "raw": raw_text,
        }
    return parsed


def _analyze_with_llm_in_worker(payload: dict[str, Any]) -> dict[str, Any]:
    transcript_blocks = list(payload["transcript_blocks"])
    transcript = _format_transcript(
        transcript_blocks,
        max_chars=int(payload["max_transcript_chars"]),
    )
    prompt = ANALYSIS_PROMPT.format(transcript=transcript)
    runtime = str(payload.get("llm_runtime", "llama_cpp")).strip().lower()
    max_tokens = int(payload["llama_max_tokens"])

    if runtime == "hf_inference":
        runner = lambda temperature: _run_hf_inference_json(
            payload,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        model_name = str(payload["hf_inference_model"])
    elif runtime == "transformers_cuda":
        model, tokenizer = _load_transformers_model(str(payload["transformers_model"]))
        runner = lambda temperature: _run_transformers_json(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        model_name = str(payload["transformers_model"])
    elif runtime == "mlx":
        model, tokenizer = _load_mlx_model(str(payload["mlx_model_path"]))
        runner = lambda temperature: _run_mlx_json(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        model_name = str(payload["mlx_model_path"])
    elif runtime == "llama_cpp":
        llm = _load_llama_cpp(
            str(payload["model_path"]),
            int(payload["llama_ctx"]),
            int(payload["llama_threads"]),
            int(payload["llama_gpu_layers"]),
        )
        runner = lambda temperature: _run_llama_cpp_json(
            llm,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        model_name = str(payload["model_name"])
    else:
        raise RuntimeError("VCA_LLM_RUNTIME must be one of: mlx, llama_cpp, hf_inference, transformers_cuda")

    analysis = runner(0.3)

    if "error" in analysis:
        retry = runner(0.0)
        if "error" in retry:
            return {
                "error": "parse_failed",
                "raw": retry.get("raw") or analysis.get("raw", ""),
            }
        analysis = retry

    analysis = enrich_analysis_from_transcript(analysis, transcript_blocks)
    analysis["analysis_mode"] = "llm"
    analysis["model"] = model_name
    return analysis


def _llm_worker_entry(payload: dict[str, Any], result_queue: multiprocessing.Queue) -> None:
    try:
        analysis = _analyze_with_llm_in_worker(payload)
        if "error" in analysis:
            result_queue.put(
                {
                    "ok": False,
                    "error": str(analysis.get("error", "unknown Phi-3 error")),
                    "raw": analysis.get("raw", ""),
                }
            )
            return

        result_queue.put(
            {
                "ok": True,
                "analysis": analysis,
            }
        )
    except Exception as exc:  # noqa: BLE001 - worker errors must be returned to parent safely.
        result_queue.put(
            {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
            }
        )


def _fallback_analysis(blocks: list[dict], mode: str, warning: str) -> dict[str, Any]:
    analysis = heuristic_analysis(blocks)
    analysis["analysis_mode"] = mode
    analysis["model"] = "rule-based fallback"
    analysis.setdefault("pipeline_warnings", []).append(warning)
    return analysis


def _raise_or_fallback(blocks: list[dict], mode: str, warning: str, allow_fallback: bool) -> dict[str, Any]:
    if allow_fallback:
        return _fallback_analysis(blocks, mode, warning)
    strict_message = warning.replace("; heuristic fallback used.", ".").replace(" heuristic fallback used.", "")
    raise RuntimeError(strict_message)


def _terminate_process(process: multiprocessing.Process) -> None:
    if not process.is_alive():
        return
    process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=2)


def _analyze_with_llm(
    transcript_blocks: list[dict],
    settings: Settings,
    *,
    allow_fallback: bool,
) -> dict[str, Any]:
    payload = {
        "transcript_blocks": transcript_blocks,
        "llm_runtime": settings.llm_runtime,
        "model_path": str(settings.phi3_model_path),
        "model_name": settings.phi3_model_path.name,
        "mlx_model_path": settings.mlx_model_path,
        "transformers_model": settings.transformers_model,
        "hf_inference_model": settings.hf_inference_model,
        "hf_inference_provider": settings.hf_inference_provider,
        "hf_inference_timeout_seconds": settings.hf_inference_timeout_seconds,
        "hf_token": settings.hf_token,
        "max_transcript_chars": settings.max_transcript_chars,
        "llama_ctx": settings.llama_ctx,
        "llama_threads": settings.llama_threads,
        "llama_gpu_layers": settings.llama_gpu_layers,
        "llama_max_tokens": settings.llama_max_tokens,
    }

    if settings.llm_runtime.strip().lower() == "transformers_cuda":
        try:
            analysis = _analyze_with_llm_in_worker(payload)
        except Exception as exc:  # noqa: BLE001 - hosted strict mode should surface runtime failures.
            return _raise_or_fallback(
                transcript_blocks,
                "llm-error-fallback",
                f"Phi-3 analysis failed: {exc}; heuristic fallback used.",
                allow_fallback,
            )

        if "error" not in analysis:
            return analysis

        if analysis.get("error") == "parse_failed":
            return _raise_or_fallback(
                transcript_blocks,
                "llm-parse-failure-fallback",
                "Phi-3 returned invalid JSON; heuristic fallback used.",
                allow_fallback,
            )

        return _raise_or_fallback(
            transcript_blocks,
            "llm-error-fallback",
            f"Phi-3 analysis failed: {analysis.get('error', 'unknown Phi-3 error')}; heuristic fallback used.",
            allow_fallback,
        )

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_llm_worker_entry,
        args=(payload, result_queue),
        name="vca-phi3-analyzer",
    )
    process.start()
    process.join(timeout=max(1, settings.llama_timeout_seconds))

    if process.is_alive():
        _terminate_process(process)
        result_queue.close()
        return _raise_or_fallback(
            transcript_blocks,
            "llm-timeout-fallback",
            f"Phi-3 analysis timed out after {settings.llama_timeout_seconds}s; heuristic fallback used.",
            allow_fallback,
        )

    try:
        result = result_queue.get(timeout=2)
    except queue.Empty:
        return _raise_or_fallback(
            transcript_blocks,
            "llm-error-fallback",
            f"Phi-3 worker exited without a result; heuristic fallback used. Exit code: {process.exitcode}",
            allow_fallback,
        )
    finally:
        result_queue.close()

    if result.get("ok"):
        analysis = result.get("analysis", {})
        if isinstance(analysis, dict):
            return analysis

    error = str(result.get("error", "unknown Phi-3 error"))
    if error == "parse_failed":
        return _raise_or_fallback(
            transcript_blocks,
            "llm-parse-failure-fallback",
            "Phi-3 returned invalid JSON; heuristic fallback used.",
            allow_fallback,
        )

    return _raise_or_fallback(
        transcript_blocks,
        "llm-error-fallback",
        f"Phi-3 analysis failed: {error}; heuristic fallback used.",
        allow_fallback,
    )


def analyze_transcript(
    transcript_blocks: list[dict],
    settings: Settings,
    analyzer_mode: str | None = None,
) -> dict[str, Any]:
    mode = (analyzer_mode or settings.analyzer_mode).lower()
    runtime = settings.llm_runtime.strip().lower()

    if mode not in {"auto", "llm", "heuristic"}:
        raise ValueError("analyzer_mode must be one of: auto, llm, heuristic")

    if runtime not in {"mlx", "llama_cpp", "hf_inference", "transformers_cuda"}:
        raise ValueError("VCA_LLM_RUNTIME must be one of: mlx, llama_cpp, hf_inference, transformers_cuda")

    llm_configured = (
        runtime == "mlx"
        or (runtime == "hf_inference" and bool(settings.hf_token))
        or (runtime == "transformers_cuda" and bool(settings.hf_token))
        or settings.phi3_model_path.exists()
    )

    if mode == "auto" and llm_configured:
        try:
            return _analyze_with_llm(transcript_blocks, settings, allow_fallback=True)
        except Exception as exc:  # noqa: BLE001 - auto mode should keep the demo path usable.
            analysis = heuristic_analysis(transcript_blocks)
            analysis["analysis_mode"] = "heuristic-auto-fallback"
            analysis["model"] = "rule-based fallback"
            analysis.setdefault("pipeline_warnings", []).append(
                f"Phi-3 analysis fallback used: {exc}"
            )
            return analysis

    if mode == "llm" and llm_configured:
        return _analyze_with_llm(transcript_blocks, settings, allow_fallback=False)

    if mode == "llm":
        raise FileNotFoundError(
            "Phi-3 is not configured. Set VCA_LLM_RUNTIME=hf_inference with HF_TOKEN, "
            "set VCA_LLM_RUNTIME=transformers_cuda with HF_TOKEN for hosted Modal GPU Phi-3, "
            "set VCA_LLM_RUNTIME=mlx to use the local MLX Phi-3 model, download the "
            f"GGUF file at {settings.phi3_model_path}, or use --analyzer-mode heuristic."
        )

    analysis = heuristic_analysis(transcript_blocks)
    analysis["analysis_mode"] = "heuristic-dev-fallback"
    analysis["model"] = "rule-based fallback"
    return analysis


def heuristic_analysis(blocks: list[dict]) -> dict[str, Any]:
    joined = " ".join(str(block.get("text", "")) for block in blocks).lower()
    agent_text = " ".join(str(block.get("text", "")) for block in blocks if block.get("speaker") == "Agent").lower()
    customer_text = " ".join(str(block.get("text", "")) for block in blocks if block.get("speaker") == "Customer").lower()

    negative_words = ("frustrated", "angry", "upset", "problem", "issue", "wrong", "not working", "charged", "cancel")
    positive_words = ("thank", "thanks", "great", "resolved", "helpful", "appreciate", "perfect")

    negative_count = sum(joined.count(word) for word in negative_words)
    positive_count = sum(joined.count(word) for word in positive_words)

    sentiment = "mixed"
    if positive_count > negative_count + 1:
        sentiment = "positive"
    elif negative_count > positive_count + 1:
        sentiment = "negative"
    elif positive_count == 0 and negative_count == 0:
        sentiment = "neutral"

    greeting_score = 8 if any(cue in agent_text for cue in ("thank you for calling", "how can i help", "my name is")) else 5
    listening_score = 8 if any(cue in agent_text for cue in ("understand", "i see", "let me check", "sorry")) else 5
    resolution_score = 8 if any(cue in agent_text for cue in ("resolved", "fixed", "refund", "credit", "next step")) else 5
    professionalism_score = 8 if any(cue in agent_text for cue in ("please", "thank", "sorry", "happy to help")) else 6
    closing_score = 8 if any(cue in agent_text for cue in ("anything else", "have a great", "thank you for calling")) else 5

    score_breakdown = {
        "greeting_and_opening": greeting_score,
        "active_listening": listening_score,
        "problem_resolution": resolution_score,
        "professionalism": professionalism_score,
        "closing": closing_score,
    }
    agent_score = round(sum(score_breakdown.values()) / len(score_breakdown))

    first_customer = next((b for b in blocks if b.get("speaker") == "Customer"), None)
    last_customer = next((b for b in reversed(blocks) if b.get("speaker") == "Customer"), None)
    first_agent = next((b for b in blocks if b.get("speaker") == "Agent"), None)

    summary_issue = _shorten(first_customer.get("text", "") if first_customer else "The customer described their issue.")
    summary_resolution = _shorten(last_customer.get("text", "") if last_customer else "The call ended without enough detail.")

    return normalize_analysis_schema(
        {
            "call_summary": (
                f"The customer contacted support about: {summary_issue} "
                f"By the end of the call, the customer response was: {summary_resolution}"
            ),
            "overall_sentiment": sentiment,
            "customer_sentiment_journey": _sentiment_journey(sentiment, customer_text),
            "agent_score": agent_score,
            "score_breakdown": score_breakdown,
            "strengths": _fallback_strengths(first_agent, listening_score, resolution_score),
            "improvement_areas": _fallback_improvements(score_breakdown),
            "recommended_next_steps": [
                "Use a clear ownership statement early in the call, such as 'I can help with that today.'",
                "Summarize the customer's issue before moving into troubleshooting or policy details.",
                "Close with the resolution, expected timeline, and one final check for remaining questions.",
            ],
            "key_moments": _fallback_key_moments(blocks),
            "compliance_flags": _fallback_compliance_flags(agent_text),
        }
    )


def _shorten(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _sentiment_journey(sentiment: str, customer_text: str) -> str:
    if "thank" in customer_text or "appreciate" in customer_text:
        return "Concerned at the start, then more reassured by the end."
    if sentiment == "negative":
        return "The customer remained frustrated or concerned during the call."
    if sentiment == "positive":
        return "The customer sounded positive or satisfied overall."
    return "The customer's mood appears mostly neutral with limited emotional change."


def _fallback_strengths(first_agent: dict | None, listening_score: int, resolution_score: int) -> list[str]:
    strengths = []
    if first_agent:
        strengths.append(
            f"Opened the interaction with a clear agent turn around {format_timestamp(first_agent.get('start', 0))}: "
            f"'{_shorten(str(first_agent.get('text', '')), 120)}'"
        )
    if listening_score >= 8:
        strengths.append("Used acknowledgement language that signals active listening.")
    if resolution_score >= 8:
        strengths.append("Provided a resolution-oriented response instead of leaving the issue open-ended.")
    return strengths or ["Maintained a usable call flow with enough information for follow-up coaching."]


def _fallback_improvements(score_breakdown: dict[str, int]) -> list[str]:
    improvements = []
    if score_breakdown["greeting_and_opening"] < 7:
        improvements.append("Strengthen the opening with name, company, and a direct offer to help.")
    if score_breakdown["active_listening"] < 7:
        improvements.append("Add a brief recap of the customer's issue to prove understanding before solving.")
    if score_breakdown["problem_resolution"] < 7:
        improvements.append("State the resolution or next action more explicitly, including owner and timeline.")
    if score_breakdown["closing"] < 7:
        improvements.append("End with a concise recap and ask whether the customer needs anything else.")
    return improvements or ["Continue making feedback specific by tying each coaching point to a timestamp."]


def _fallback_key_moments(blocks: list[dict]) -> list[dict]:
    moments = []
    for block in blocks[:3]:
        start = format_timestamp(block.get("start", 0))
        end = format_timestamp(block.get("end", 0))
        moments.append(
            {
                "timestamp_range": f"{start}-{end}",
                "speaker": block.get("speaker", "Speaker"),
                "description": _shorten(str(block.get("text", "")), 140),
                "impact": "neutral",
            }
        )
    return moments


def _fallback_compliance_flags(agent_text: str) -> list[str]:
    if any(cue in agent_text for cue in ("verify", "account number", "security")):
        return ["No compliance issues detected"]
    return ["Account verification was not clearly present in the transcript; verify whether this was required for the call type."]
