from pathlib import Path

import yaml
import requests

# The project root is the folder above src/, wherever the terminal happens to be
ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read the pipeline settings from config.yaml."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def probe(config: dict) -> dict:
    """Ask the API for one record and return the total count and last update date."""
    api = config["api"]
    response = requests.get(
        api["base_url"],
        params={"limit": 1},
        timeout=api["timeout_seconds"],
    )
    response.raise_for_status()
    meta = response.json()["meta"]
    return {
        "total": meta["results"]["total"],
        "last_updated": meta["last_updated"],
    }


if __name__ == "__main__":
    config = load_config()
    info = probe(config)
    print(f"Records available: {info['total']:,}")
    print("Last updated:", info["last_updated"])