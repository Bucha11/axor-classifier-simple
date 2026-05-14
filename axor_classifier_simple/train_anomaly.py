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
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    min_accuracy: float = 0.90,
) -> None:
    """
    Generate synthetic anomaly data, train GradientBoostingClassifier, evaluate, and save.

    Raises RuntimeError if validation accuracy is below min_accuracy.
    """
    try:
        import joblib
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
    except ImportError as e:
        raise ImportError(
            "scikit-learn and joblib are required. "
            "Install with: pip install axor-classifier-simple[ml]"
        ) from e

    from axor_core.contracts.anomaly import NormalizedIntent
    from axor_classifier_simple.data.anomaly_data import generate
    from axor_classifier_simple.anomaly_detector import window_to_feature_vector

    out_path = Path(model_path) if model_path else _default_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic anomaly data...", flush=True)
    data = generate(seed=seed)

    X = []
    y = []
    for window_dicts, label in data:
        window = [NormalizedIntent(**d) for d in window_dicts]
        X.append(window_to_feature_vector(window))
        y.append(label)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_split, random_state=seed, stratify=y
    )

    print(f"Training GradientBoostingClassifier ({len(X_train)} train / {len(X_val)} val)...", flush=True)
    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_val, clf.predict(X_val))
    print(f"Validation accuracy: {acc:.3f}")
    if acc < min_accuracy:
        raise RuntimeError(
            f"Anomaly detector validation accuracy {acc:.3f} < required {min_accuracy:.3f}"
        )

    joblib.dump(clf, out_path)
    print(f"Model saved to {out_path}")


def _default_path() -> Path:
    return Path.home() / ".axor" / "models" / "anomaly_detector.joblib"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLAnomalyDetector")
    parser.add_argument("--out", type=Path, default=None, help="Output model path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--min-accuracy", type=float, default=0.90)
    args = parser.parse_args()
    train(
        model_path=args.out,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_accuracy=args.min_accuracy,
    )
