import json
from pathlib import Path
import re
import pandas as pd

from extract import ROOT, load_config


def latest_run_folder(config: dict) -> Path:
    """Return the newest raw run folder whose manifest says the extract was complete."""
    raw = ROOT / config["paths"]["raw"]
    runs = []
    for folder in raw.iterdir():
        manifest = folder / "manifest.json"
        if folder.is_dir() and manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")).get("complete"):
                runs.append(folder)
    if not runs:
        raise FileNotFoundError("No complete extract runs in data/raw. Run extract.py first.")
    return sorted(runs)[-1]


def load_raw(folder: Path) -> pd.DataFrame:
    """Combine every saved page in a run folder into one table."""
    records = []
    for page in sorted(folder.glob("page_*.json")):
        records.extend(json.loads(page.read_text(encoding="utf-8"))["results"])
    return pd.DataFrame(records)

DROP_COLUMNS = ["openfda", "code_info", "more_code_info"]
PLACEHOLDERS = {"", "N/A", "NA", "n/a", "None", "NONE", "Unknown", "UNKNOWN"}
DATE_COLUMNS = ["recall_initiation_date", "center_classification_date",
                "report_date", "termination_date"]
CLASS_MAP = {"Class I": "I", "Class II": "II", "Class III": "III"}
CATEGORY_COLUMNS = ["classification", "status", "voluntary_mandated",
                    "initial_firm_notification", "product_type"]


def drop_unused(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns not needed downstream (empty for food, or long lot-code lists)."""
    return df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])


def standardise_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Trim text and turn placeholder values such as '' or 'N/A' into true missing values."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()
            df[col] = df[col].where(~df[col].isin(PLACEHOLDERS))
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert YYYYMMDD strings to dates; anything unparseable becomes missing and is reported."""
    df = df.copy()
    for col in DATE_COLUMNS:
        original = df[col]
        df[col] = pd.to_datetime(original, format="%Y%m%d", errors="coerce")
        failed = original.notna() & df[col].isna()
        if failed.any():
            print(f"{col}: {failed.sum()} values could not be parsed")
    return df


def normalise_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Short class labels, upper-case states, category types and an is_terminated flag."""
    df = df.copy()
    df["classification"] = df["classification"].map(CLASS_MAP)  # 'Not Yet Classified' -> missing
    df["state"] = df["state"].str.upper()
    for col in CATEGORY_COLUMNS:
        df[col] = df[col].astype("category")
    df["is_terminated"] = df["termination_date"].notna()
    return df


US_STATES = {
    "AL": "alabama", "AK": "alaska", "AZ": "arizona", "AR": "arkansas", "CA": "california",
    "CO": "colorado", "CT": "connecticut", "DE": "delaware", "FL": "florida", "GA": "georgia",
    "HI": "hawaii", "ID": "idaho", "IL": "illinois", "IN": "indiana", "IA": "iowa",
    "KS": "kansas", "KY": "kentucky", "LA": "louisiana", "ME": "maine", "MD": "maryland",
    "MA": "massachusetts", "MI": "michigan", "MN": "minnesota", "MS": "mississippi",
    "MO": "missouri", "MT": "montana", "NE": "nebraska", "NV": "nevada", "NH": "new hampshire",
    "NJ": "new jersey", "NM": "new mexico", "NY": "new york", "NC": "north carolina",
    "ND": "north dakota", "OH": "ohio", "OK": "oklahoma", "OR": "oregon", "PA": "pennsylvania",
    "RI": "rhode island", "SC": "south carolina", "SD": "south dakota", "TN": "tennessee",
    "TX": "texas", "UT": "utah", "VT": "vermont", "VA": "virginia", "WA": "washington",
    "WV": "west virginia", "WI": "wisconsin", "WY": "wyoming", "DC": "district of columbia",
}


def keyword_pattern(keywords: list[str]) -> re.Pattern:
    """One case-insensitive pattern matching any keyword at the start of a word."""
    return re.compile(r"\b(?:" + "|".join(re.escape(k) for k in keywords) + ")", re.IGNORECASE)


def normalise_text(text) -> str | None:
    """Lower-case and collapse runs of whitespace; keep missing as missing."""
    if not isinstance(text, str):
        return None
    return re.sub(r"\s+", " ", text).strip().lower()


def add_hazard(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Derive a hazard category from the recall reason using the ordered rules in config."""
    df = df.copy()
    df["reason_clean"] = df["reason_for_recall"].map(normalise_text)
    rules = [(rule["name"], keyword_pattern(rule["keywords"])) for rule in config["hazards"]]

    def first_match(reason):
        if reason is None:
            return "unknown"
        for name, pattern in rules:
            if pattern.search(reason):
                return name
        return "other"

    df["hazard"] = df["reason_clean"].map(first_match).astype("category")
    return df


def count_states(text) -> int:
    """Count distinct US states named in a distribution pattern, by full name or 2-letter code."""
    if not isinstance(text, str):
        return 0
    found = {code for code in re.findall(r"\b[A-Z]{2}\b", text) if code in US_STATES}
    lower = text.lower()
    found |= {code for code, name in US_STATES.items() if re.search(rf"\b{name}\b", lower)}
    return len(found)


def add_distribution_flags(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Nationwide and international flags, and the number of states named, from free text."""
    df = df.copy()
    dist = df["distribution_pattern"].fillna("")
    df["nationwide"] = dist.str.contains(keyword_pattern(config["distribution"]["nationwide"]))
    df["international"] = dist.str.contains(keyword_pattern(config["distribution"]["international"]))
    df["n_states_named"] = dist.map(count_states)
    return df

def clean(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Run the cleaning steps in order."""
    df = drop_unused(df)
    df = standardise_missing(df)
    df = parse_dates(df)
    df = normalise_categories(df)
    df = add_hazard(df, config)
    df = add_distribution_flags(df, config)
    return df

def save_transformed(df: pd.DataFrame, config: dict) -> Path:
    """Write the transformed table as Parquet, which keeps dates, categories and flags typed."""
    out_dir = ROOT / config["paths"]["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "recalls_transformed.parquet"
    df.to_parquet(path, index=False)
    return path

if __name__ == "__main__":
    config = load_config()
    folder = latest_run_folder(config)
    raw = load_raw(folder)
    df = clean(raw, config)
    path = save_transformed(df, config)
    print(f"Saved transformed table to {path}")

    print(f"Cleaned {len(df):,} rows, {df.shape[1]} columns\n")
    print("Missing after cleaning:\n", df.isna().sum()[lambda s: s > 0], "\n")
    print("Classification:\n", df["classification"].value_counts(dropna=False), "\n")
    print("Status vs is_terminated:\n", pd.crosstab(df["status"], df["is_terminated"]), "\n")
    print("Report dates from", df["report_date"].min().date(), "to", df["report_date"].max().date())
    late = (df["recall_initiation_date"] > df["center_classification_date"]).sum()
    print("Recalls classified before they started (suspicious):", late)
    print("\nHazard categories:\n", df["hazard"].value_counts())
    print(f"\nShare in 'other': {(df['hazard'] == 'other').mean():.1%}")
    print("\nHazard by class (row %):\n",
          pd.crosstab(df["hazard"], df["classification"], normalize="index").round(2))
    print("\nNationwide:", df["nationwide"].sum(), " International:", df["international"].sum())
    print("States named (summary):\n", df["n_states_named"].describe().round(1))
    print("\nTen 'other' reasons to inspect:")
    for reason in df.loc[df["hazard"] == "other", "reason_clean"].sample(10, random_state=1):
        print(" -", reason[:150])
    print("\nFive 'international' examples to check:")
    for text in df.loc[df["international"], "distribution_pattern"].sample(5, random_state=1):
        print(" -", text[:150])

