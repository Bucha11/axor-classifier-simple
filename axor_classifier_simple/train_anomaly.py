"""
Train and serialize the MLAnomalyDetector model.

Usage:
    python -m axor_classifier_simple.train_anomaly
    python -m axor_classifier_simple.train_anomaly --out /path/to/model.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path


def train(
    model_path: Path | str | None = None,
    seed: int = 42,
    test_split: float = 0.15,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    min_synthetic_accuracy: float = 0.90,
    min_hard_accuracy: float = 0.75,
) -> dict[str, float]:
    """
    Generate synthetic anomaly data, train GradientBoostingClassifier, evaluate, and save.

    Two accuracy metrics are reported:
        synthetic val  — held-out slice of the template-generated corpus.
                         Expected: ~99%+. High accuracy here is easy; don't over-index on it.
        hard eval      — boundary cases not derived from any training builder.
                         This is the real quality signal. Target: 80%+.

    Raises RuntimeError if either accuracy gate is not met.
    Returns {"synthetic_acc": float, "hard_acc": float}.
    """
    try:
        import joblib
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError as e:
        raise ImportError(
            "scikit-learn and joblib are required. "
            "Install with: pip install axor-classifier-simple[ml]"
        ) from e

    from axor_core.contracts.anomaly import NormalizedIntent
    from axor_classifier_simple.data.anomaly_data import LINEAGE_KEYS, generate, generate_hard

    def _to_intent(d: dict) -> NormalizedIntent:
        # Corpus dicts carry per-value lineage annotations (e.g. carries_secret)
        # that are not NormalizedIntent fields — strip them before reconstructing.
        return NormalizedIntent(**{k: v for k, v in d.items() if k not in LINEAGE_KEYS})
    from axor_classifier_simple.anomaly_detector import window_to_feature_vector

    out_path = Path(model_path) if model_path else _default_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic training data...", flush=True)
    data = generate(seed=seed)
    print(f"  {len(data)} windows")

    X = []
    y = []
    for window_dicts, label in data:
        window = [_to_intent(d) for d in window_dicts]
        X.append(window_to_feature_vector(window))
        y.append(label)

    print("Loading hard eval set...", flush=True)
    hard_data = generate_hard(seed=seed)
    X_hard = []
    y_hard = []
    for window_dicts, label in hard_data:
        window = [_to_intent(d) for d in window_dicts]
        X_hard.append(window_to_feature_vector(window))
        y_hard.append(label)
    print(f"  {len(X_hard)} hard eval examples (boundary cases, not used for training)")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y
    )

    print(f"\nTraining GradientBoostingClassifier ({len(X_train)} train / {len(X_val)} val)...", flush=True)
    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    syn_acc  = accuracy_score(y_val,    clf.predict(X_val))
    hard_acc = accuracy_score(y_hard,   clf.predict(X_hard))

    print(f"  synthetic val : {syn_acc:.3f}")
    print(f"  hard eval     : {hard_acc:.3f}  ← real quality signal")
    print()
    print(classification_report(y_hard, clf.predict(X_hard),
                                 target_names=["critical", "normal", "suspicious"],
                                 labels=["critical", "normal", "suspicious"]))

    if syn_acc < min_synthetic_accuracy:
        raise RuntimeError(
            f"Synthetic val accuracy {syn_acc:.3f} < required {min_synthetic_accuracy:.3f}"
        )
    if hard_acc < min_hard_accuracy:
        raise RuntimeError(
            f"Hard eval accuracy {hard_acc:.3f} < required {min_hard_accuracy:.3f}"
        )

    joblib.dump(clf, out_path)
    print(f"Model saved to {out_path}")

    return {"synthetic_acc": syn_acc, "hard_acc": hard_acc}


def _default_path() -> Path:
    return Path.home() / ".axor" / "models" / "anomaly_detector.joblib"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLAnomalyDetector")
    parser.add_argument("--out", type=Path, default=None, help="Output model path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--min-synthetic-accuracy", type=float, default=0.90)
    parser.add_argument("--min-hard-accuracy",      type=float, default=0.75)
    args = parser.parse_args()
    train(
        model_path=args.out,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_synthetic_accuracy=args.min_synthetic_accuracy,
        min_hard_accuracy=args.min_hard_accuracy,
    )
