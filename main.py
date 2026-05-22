from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.pipeline import run_pipeline, run_transcript_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="AI-powered voice call analysis pipeline")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Path to .mp3, .wav, or .m4a call recording")
    source.add_argument("--transcript-file", type=Path, help="Path to a labeled transcript for smoke testing")
    parser.add_argument("--output", type=Path, help="Output directory for report files")
    parser.add_argument(
        "--analyzer-mode",
        choices=("auto", "llm", "heuristic"),
        default="auto",
        help="Use Phi-3 when available, require Phi-3, or run the dev fallback analyzer",
    )
    parser.add_argument("--json", action="store_true", help="Print final result metadata as JSON")
    args = parser.parse_args()

    def progress(stage: str, percent: int, message: str) -> None:
        print(f"[{percent:3d}%] {stage}: {message}")

    if args.transcript_file:
        result = run_transcript_pipeline(
            args.transcript_file,
            output_dir=args.output,
            analyzer_mode=args.analyzer_mode,
            progress_callback=progress,
        )
    else:
        result = run_pipeline(
            args.input,
            output_dir=args.output,
            analyzer_mode=args.analyzer_mode,
            progress_callback=progress,
        )

    print()
    print(f"Report ID: {result['id']}")
    print(f"JSON report: {result['paths']['json']}")
    print(f"Markdown report: {result['paths']['markdown']}")
    print(f"Transcript: {result['paths']['transcript']}")

    if args.json:
        print(json.dumps({"id": result["id"], "paths": result["paths"]}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

