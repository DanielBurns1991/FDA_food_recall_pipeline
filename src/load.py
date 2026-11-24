import json
import subprocess
from pathlib import Path

from extract import ROOT, load_config

WARN_MB, LIMIT_MB = 50, 100  # GitHub warns above 50 MB and rejects files above 100 MB


def git(*args: str) -> str:
    """Run a git command in the project folder and return its output; fail loudly on errors."""
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def check_sizes(paths: list[Path]) -> None:
    """Stop before GitHub would reject a file, and warn when one is getting large."""
    for path in paths:
        mb = path.stat().st_size / 1e6
        if mb > LIMIT_MB:
            raise RuntimeError(f"{path.name} is {mb:.1f} MB; GitHub rejects files over {LIMIT_MB} MB")
        if mb > WARN_MB:
            print(f"Warning: {path.name} is {mb:.1f} MB (GitHub warns above {WARN_MB} MB)")


def commit_message(config: dict) -> str:
    """Describe the data being loaded: source run, clean row count and FDA's update date."""
    out = ROOT / config["paths"]["processed"]
    quality = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
    manifest_path = ROOT / config["paths"]["raw"] / quality["source_run"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (f"Data refresh {quality['source_run']}: {quality['rows_clean']:,} clean rows, "
            f"{quality['rows_quarantined']} quarantined, FDA data updated "
            f"{manifest['source_last_updated']}")


def load(config: dict) -> bool:
    """Commit and push the processed outputs if they changed. Returns True if a commit was made."""
    paths = [ROOT / f for f in config["load"]["files"]]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Run the earlier stages first; missing: {missing}")
    check_sizes(paths)

    relative = [p.relative_to(ROOT).as_posix() for p in paths]
    git("add", "--", *relative)
    if not git("diff", "--cached", "--name-only", "--", *relative):
        print("Nothing to load: the processed files match the last commit.")
        return False

    message = commit_message(config)
    git("commit", "-m", message, "--", *relative)
    print(f"Committed: {message}")
    if config["load"]["push"]:
        git("push")
        print("Pushed to GitHub.")
    return True


if __name__ == "__main__":
    load(load_config())