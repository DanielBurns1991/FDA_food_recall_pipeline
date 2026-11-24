import pandas as pd
import json
from datetime import datetime, timezone
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from extract import ROOT, load_config

TARGET = "classification"
GROUP = "event_id"

# Fields created after FDA classifies a recall, or that identify the firm. Never used as features.
EXCLUDED = {
    "center_classification_date": "exists only after FDA classifies the recall",
    "report_date": "the enforcement report is published after classification",
    "recall_number": "FDA assigns it to a product once it is classified",
    "status": "describes the outcome months later",
    "termination_date": "describes the outcome months later",
    "is_terminated": "derived from termination_date",
    "recalling_firm": "lets the model memorise firms, and is unfair to them",
}

TEXT = "reason_clean"
CATEGORICAL = ["hazard", "voluntary_mandated", "initial_firm_notification", "state"]
BINARY = ["nationwide", "international"]
NUMERIC = ["n_states_named"]
FEATURES = [TEXT] + CATEGORICAL + BINARY + NUMERIC


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Select only the allowed features and make missing values explicit."""
    X = df[FEATURES].copy()
    X[TEXT] = X[TEXT].fillna("")
    for col in CATEGORICAL:
        X[col] = X[col].astype("object").fillna("missing").astype(str)
    X[BINARY] = X[BINARY].astype(int)
    return X


def make_preprocessor() -> ColumnTransformer:
    """Turn text into TF-IDF weights, categories into one-hot columns, and scale the count."""
    return ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True), TEXT),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("bin", "passthrough", BINARY),
        ("num", StandardScaler(), NUMERIC),
    ])


def make_models(config: dict) -> dict:
    """The two candidate models, each a full pipeline including its preprocessing."""
    seed = config["model"]["random_seed"]
    return {
        "Logistic regression": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]),
        "Random forest": Pipeline([
            ("prep", make_preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                           class_weight="balanced", n_jobs=-1, random_state=seed)),
        ]),
    }


def load_training_data(config: dict) -> pd.DataFrame:
    """Clean recalls that have a known class (I, II or III)."""
    df = pd.read_parquet(ROOT / config["paths"]["processed"] / "recalls_clean.parquet")
    df = df[df[TARGET].notna()].copy()
    df[TARGET] = df[TARGET].astype(str)
    return df


def split_by_event(df: pd.DataFrame, config: dict):
    """Keep each recall event on one side of the split, while keeping class shares similar."""
    n_splits = round(1 / config["model"]["test_size"])  # 0.2 -> 5 folds; one fold is the test set
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                    random_state=config["model"]["random_seed"])
    train_idx, test_idx = next(splitter.split(df, df[TARGET], groups=df[GROUP]))
    return df.iloc[train_idx], df.iloc[test_idx]


def evaluate(name: str, y_true: pd.Series, y_pred) -> dict:
    """The three scores we report for every model."""
    return {
        "model": name,
        "accuracy": round(float((y_true == y_pred).mean()), 3),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 3),
        "class_I_recall": round(recall_score(y_true, y_pred, labels=["I"], average="macro"), 3),
    }


def keyword_baseline(df: pd.DataFrame, config: dict) -> pd.Series:
    """Pathogen hazards predict Class I; everything else Class II."""
    pathogens = set(config["model"]["pathogen_hazards"])
    return df["hazard"].astype(str).map(lambda h: "I" if h in pathogens else "II")

def top_terms(pipeline: Pipeline, n: int = 12) -> pd.DataFrame:
    """The features that push the logistic regression most strongly towards each class."""
    names = pipeline.named_steps["prep"].get_feature_names_out()
    clf = pipeline.named_steps["clf"]
    columns = {}
    for i, label in enumerate(clf.classes_):
        strongest = clf.coef_[i].argsort()[::-1][:n]
        columns[f"Class {label}"] = [names[j].split("__", 1)[1] for j in strongest]
    return pd.DataFrame(columns)

def save_outputs(best: str, evaluated: Pipeline, final: Pipeline, scores: pd.DataFrame,
                 train: pd.DataFrame, test: pd.DataFrame, config: dict) -> None:
    """Save the deployable model, test-set predictions, metrics and predictions for pending recalls."""
    out = ROOT / config["paths"]["processed"]
    model_dir = ROOT / config["paths"]["models"]
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, model_dir / "recall_classifier.joblib")

    # Test-set predictions from the model that never saw these events
    proba = evaluated.predict_proba(features(test))
    preds = pd.DataFrame({
        "recall_number": test["recall_number"].values,
        "event_id": test[GROUP].values,
        "actual": test[TARGET].values,
        "predicted": evaluated.classes_[proba.argmax(axis=1)],
    })
    for i, label in enumerate(evaluated.classes_):
        preds[f"prob_{label}"] = proba[:, i].round(4)
    preds.to_csv(out / "predictions.csv", index=False)

    labels = ["I", "II", "III"]
    metrics = {
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_model": best,
        "selection_metric": config["model"]["selection_metric"],
        "split": "StratifiedGroupKFold on event_id, one of five folds held out",
        "random_seed": config["model"]["random_seed"],
        "train_rows": len(train),
        "test_rows": len(test),
        "features": FEATURES,
        "excluded_features": EXCLUDED,
        "scores": scores.to_dict(orient="records"),
        "confusion_matrix": {"labels": labels,
                             "matrix": confusion_matrix(preds["actual"], preds["predicted"],
                                                        labels=labels).tolist()},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Recalls FDA has not classified yet: the model's real use case
    pending = pd.read_csv(out / "quarantine.csv")
    pending = pending[pending[TARGET].isna()]
    if len(pending):
        p = final.predict_proba(features(pending))
        pending = pending.assign(predicted=final.classes_[p.argmax(axis=1)],
                                 prob_I=p[:, list(final.classes_).index("I")].round(3))
        pending[["event_id", "reason_for_recall", "predicted", "prob_I"]].to_csv(
            out / "pending_predictions.csv", index=False)
        print("\nPredictions for recalls FDA has not classified yet:")
        print(pending[["event_id", "reason_for_recall", "predicted", "prob_I"]].to_string(index=False))

if __name__ == "__main__":
    config = load_config()
    df = load_training_data(config)
    train, test = split_by_event(df, config)

    shared = set(train[GROUP]) & set(test[GROUP])
    print(f"Train {len(train):,} rows / test {len(test):,} rows; events in both: {len(shared)}")
    print("Class share, train:", train[TARGET].value_counts(normalize=True).round(3).to_dict())
    print("Class share, test: ", test[TARGET].value_counts(normalize=True).round(3).to_dict(), "\n")

    majority = DummyClassifier(strategy="most_frequent").fit(train[[TARGET]], train[TARGET])
    results = [
        evaluate("Majority class", test[TARGET], majority.predict(test[[TARGET]])),
        evaluate("Keyword rule", test[TARGET], keyword_baseline(test, config)),
    ]
    fitted = {}
    for name, pipeline in make_models(config).items():
        print(f"Training {name}...")
        pipeline.fit(features(train), train[TARGET])
        fitted[name] = pipeline
        results.append(evaluate(name, test[TARGET], pipeline.predict(features(test))))

    scores = pd.DataFrame(results)
    print("\n", scores.to_string(index=False))

    metric = config["model"]["selection_metric"]
    best = scores.iloc[2:].sort_values([metric, "macro_f1"], ascending=False).iloc[0]["model"]
    print(f"\nSelected by {metric}: {best}")
    predictions = fitted[best].predict(features(test))
    labels = ["I", "II", "III"]
    print(classification_report(test[TARGET], predictions, labels=labels, digits=3))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(confusion_matrix(test[TARGET], predictions, labels=labels),
                       index=labels, columns=labels))
    print("\nWords and features pushing towards each class (logistic regression):")
    print(top_terms(fitted["Logistic regression"]).to_string(index=False))
    final = make_models(config)[best].fit(features(df), df[TARGET])
    save_outputs(best, fitted[best], final, scores, train, test, config)
    print(f"\nSaved model to {ROOT / config['paths']['models'] / 'recall_classifier.joblib'}")