import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGES = ["extract", "transform", "validate", "model", "load"]


def run_stage(name: str, log) -> None:
    """Run one stage script, showing and logging its output; stop the pipeline if it fails."""
    header = f"\n=== {name} ({datetime.now(timezone.utc):%H:%M:%S} UTC) ===\n"
    print(header, end="")
    log.write(header)

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    start = time.perf_counter()
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "src" / f"{name}.py")], cwd=ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
    )
    for line in process.stdout:
        print(line, end="")
        log.write(line)
    process.wait()
    seconds = time.perf_counter() - start

    if process.returncode != 0:
        message = f"\nStage '{name}' FAILED after {seconds:.0f}s. Pipeline stopped.\n"
        print(message, end="")
        log.write(message)
        sys.exit(process.returncode)
    summary = f"Stage '{name}' finished in {seconds:.0f}s\n"
    print(summary, end="")
    log.write(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FDA food recall pipeline end to end.")
    parser.add_argument("--skip-extract", action="store_true",
                        help="reuse the latest complete raw run instead of calling the API")
    parser.add_argument("--no-load", action="store_true",
                        help="run every stage except the GitHub load")
    args = parser.parse_args()

    stages = [s for s in STAGES
              if not (s == "extract" and args.skip_extract) and not (s == "load" and args.no_load)]

    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"pipeline_{datetime.now(timezone.utc):%Y-%m-%dT%H%M%SZ}.log"
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"Stages: {', '.join(stages)}\n")
        start = time.perf_counter()
        for stage in stages:
            run_stage(stage, log)
        done = f"\nPipeline complete in {time.perf_counter() - start:.0f}s. Log: {log_path}\n"
        print(done, end="")
        log.write(done)


if __name__ == "__main__":
    main()