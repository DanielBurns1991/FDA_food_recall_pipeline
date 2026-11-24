import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

# The project root is the folder above src/, wherever the terminal happens to be
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"

# Errors worth retrying: rate limit (429) and temporary server problems (5xx)
RETRY_STATUSES = {429, 500, 502, 503, 504}


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read the pipeline settings from config.yaml."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_with_retries(url: str, params: dict | None, config: dict) -> requests.Response:
    """GET a URL, retrying rate-limit and server errors with exponential back-off."""
    api = config["api"]
    for attempt in range(1, api["max_retries"] + 1):
        try:
            response = requests.get(url, params=params, timeout=api["timeout_seconds"])
        except (requests.ConnectionError, requests.Timeout) as error:
            problem = type(error).__name__
        else:
            if response.status_code not in RETRY_STATUSES:
                response.raise_for_status()
                return response
            problem = f"HTTP {response.status_code}"
        wait = 2 ** attempt
        print(f"Attempt {attempt} failed ({problem}); retrying in {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"Gave up after {api['max_retries']} attempts: {url}")


def probe(config: dict) -> dict:
    """Ask the API for one record and return the total count and last update date."""
    response = get_with_retries(config["api"]["base_url"], {"limit": 1}, config)
    meta = response.json()["meta"]
    return {
        "total": meta["results"]["total"],
        "last_updated": meta["last_updated"],
    }


def make_run_folder(config: dict) -> Path:
    """Create data/raw/<run time>/ for this run and return its path."""
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    folder = ROOT / config["paths"]["raw"] / run_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_page(text: str, folder: Path, page_number: int) -> Path:
    """Save one API response exactly as it arrived."""
    path = folder / f"page_{page_number:04d}.json"
    path.write_text(text, encoding="utf-8")
    return path

def fetch_all(config: dict, folder: Path) -> tuple[int, int]:
    """Follow the next-page links until the last page, saving each page. Returns (pages, records)."""
    api = config["api"]
    url = api["base_url"]
    params = {"limit": api["page_size"], "sort": f"{api['sort_field']}:asc"}
    pages = 0
    records = 0

    while url:
        response = get_with_retries(url, params, config)
        pages += 1
        save_page(response.text, folder, pages)
        count = len(response.json()["results"])
        records += count
        print(f"Page {pages}: {count} records (running total {records:,})")

        url = response.links.get("next", {}).get("url")
        params = None  # the next-page URL already contains every parameter
        time.sleep(api["pause_seconds"])

        if pages >= 200:  # safety stop: about 30 pages are expected
            raise RuntimeError("Stopped after 200 pages; check the paging logic")

    return pages, records

def write_manifest(folder: Path, info: dict, pages: int, fetched: int,
                   started: str, finished: str, config: dict) -> Path:
    """Record what this run did, next to the pages it saved."""
    manifest = {
        "run_id": folder.name,
        "source_url": config["api"]["base_url"],
        "started_utc": started,
        "finished_utc": finished,
        "pages": pages,
        "records_fetched": fetched,
        "records_expected": info["total"],
        "source_last_updated": info["last_updated"],
        "complete": fetched == info["total"],
    }
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def check_complete(info: dict, fetched: int, config: dict) -> None:
    """Stop the pipeline if records are missing, unless FDA updated the data mid-run."""
    if fetched == info["total"]:
        return
    latest = probe(config)
    if latest["last_updated"] != info["last_updated"]:
        print(f"Warning: FDA updated the data during the run "
              f"({fetched:,} fetched, {latest['total']:,} now available). Re-run to refresh.")
        return
    raise RuntimeError(f"Incomplete extract: fetched {fetched:,} of {info['total']:,} records")

if __name__ == "__main__":
    config = load_config()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    info = probe(config)
    print(f"Records available: {info['total']:,} (updated {info['last_updated']})")

    folder = make_run_folder(config)
    pages, fetched = fetch_all(config, folder)

    finished = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_path = write_manifest(folder, info, pages, fetched, started, finished, config)
    print(f"Manifest written to {manifest_path}")

    check_complete(info, fetched, config)
    print(f"Extract complete: {pages} pages, {fetched:,} records.")