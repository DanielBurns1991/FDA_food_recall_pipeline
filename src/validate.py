import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from extract import ROOT, load_config
from transform import DATE_COLUMNS, latest_run_folder

REQUIRED_COLUMNS = [
    "recall_number", "event_id", "classification", "reason_for_recall", "status",
    "report_date", "recall_initiation_date", "center_classification_date", "termination_date",
    "hazard", "nationwide", "international", "n_states_named", "is_terminated",
]


class ValidationError(Exception):
    """Raised when a stop rule fails: the data must not move downstream."""


@dataclass
class Rule:
    name: str
    action: str                                   # "stop", "quarantine" or "warn"
    description: str
    failing: Callable[[pd.DataFrame], pd.Series]  # True for each row that breaks the rule


def today() -> pd.Timestamp:
    return pd.Timestamp.now().normalize()


RULES = [
    Rule("recall_number_unique", "stop", "Each recall number appears only once",
         lambda df: df["recall_number"].notna() & df["recall_number"].duplicated(keep=False)),
    Rule("recall_number_present", "quarantine", "Every record has a recall number",
         lambda df: df["recall_number"].isna()),
    Rule("reason_present", "quarantine", "Every record has a reason for recall",
         lambda df: df["reason_for_recall"].isna()),
    Rule("report_date_present", "quarantine", "Every record has a report date",
         lambda df: df["report_date"].isna()),
    Rule("no_future_dates", "quarantine", "No date is later than today",
         lambda df: (df[DATE_COLUMNS] > today()).any(axis=1)),
    Rule("classification_known", "warn",
         "Classification is I, II or III (unclassified rows are kept but not used for training)",
         lambda df: df["classification"].isna()),
    Rule("terminated_has_date", "warn", "Terminated recalls have a termination date",
         lambda df: (df["status"] == "Terminated") & df["termination_date"].isna()),
    Rule("initiated_before_classified", "warn", "A recall starts on or before FDA classifies it",
         lambda df: df["recall_initiation_date"] > df["center_classification_date"]),
    Rule("us_firm_has_state", "warn", "Firms in the United States have a state",
         lambda df: (df["country"] == "United States") & df["state"].isna()),
]


def check_schema(df: pd.DataFrame) -> list[str]:
    """Structural checks: required columns exist and date columns really are dates."""
    problems = [f"missing column: {c}" for c in REQUIRED_COLUMNS if c not in df.columns]
    for col in DATE_COLUMNS:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            problems.append(f"{col} is not a date column")
    return problems


def validate(df: pd.DataFrame, config: dict):
    """Apply every rule. Returns clean rows, quarantined rows, per-rule results and stop failures."""
    schema_problems = check_schema(df)
    if schema_problems:
        raise ValidationError("Schema check failed: " + "; ".join(schema_problems))

    results, stop_failures = [], []
    quarantined = pd.Series(False, index=df.index)
    reasons = pd.Series("", index=df.index)

    for rule in RULES:
        failing = rule.failing(df).fillna(False).astype(bool)
        n = int(failing.sum())
        results.append({"rule": rule.name, "action": rule.action, "description": rule.description,
                        "checked": len(df), "failed": n, "passed": len(df) - n})
        if rule.action == "quarantine":
            quarantined |= failing
            reasons.loc[failing] = reasons.loc[failing] + rule.name + "; "
        elif rule.action == "stop" and n:
            stop_failures.append(f"{rule.name} ({n} rows)")

    share = quarantined.mean()
    limit = config["validation"]["max_quarantine_share"]
    if share > limit:
        stop_failures.append(f"quarantine share {share:.2%} is above the {limit:.0%} limit")

    clean = df[~quarantined]
    held = df[quarantined].assign(failed_rules=reasons[quarantined].str.rstrip("; "))
    return clean, held, results, stop_failures


def save_outputs(clean, held, results, stop_failures, config) -> Path:
    """Always write the quality report and quarantine file; write clean data only if nothing stopped."""
    out = ROOT / config["paths"]["processed"]
    report = {
        "validated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_run": latest_run_folder(config).name,
        "rows_in": len(clean) + len(held),
        "rows_clean": len(clean),
        "rows_quarantined": len(held),
        "passed": not stop_failures,
        "stop_failures": stop_failures,
        "rules": results,
    }
    (out / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    held.to_csv(out / "quarantine.csv", index=False, date_format="%Y-%m-%d")

    if stop_failures:
        raise ValidationError("Validation stopped the pipeline: " + "; ".join(stop_failures))

    clean_path = out / "recalls_clean.csv"
    clean.to_csv(clean_path, index=False, date_format="%Y-%m-%d")
    clean.to_parquet(out / "recalls_clean.parquet", index=False)
    return clean_path

print(df.loc[df["state"].isna(), "country"].value_counts().head(10))


if __name__ == "__main__":
    config = load_config()
    df = pd.read_parquet(ROOT / config["paths"]["processed"] / "recalls_transformed.parquet")
    clean, held, results, stop_failures = validate(df, config)

    print(pd.DataFrame(results)[["rule", "action", "failed"]].to_string(index=False), "\n")
    path = save_outputs(clean, held, results, stop_failures, config)
    print(f"{len(clean):,} clean rows saved to {path} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"{len(held):,} rows quarantined (see quarantine.csv)")