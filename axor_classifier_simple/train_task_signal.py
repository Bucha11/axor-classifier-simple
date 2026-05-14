"""
Train and serialize the TaskSignalClassifier model.

Usage:
    python -m axor_classifier_simple.train_task_signal
    python -m axor_classifier_simple.train_task_signal --out /path/to/model.joblib
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def train(
    model_path: Path | str | None = None,
    seed: int = 42,
    test_split: float = 0.15,
    min_accuracy: float = 0.85,
) -> None:
    """
    Generate synthetic data, train all three heads, evaluate, and save.

    Raises RuntimeError if validation accuracy is below min_accuracy.
    """
    try:
        import joblib
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
    except ImportError as e:
        raise ImportError(
            "scikit-learn and joblib are required. "
            "Install with: pip install axor-classifier-simple[ml]"
        ) from e

    from axor_classifier_simple.data.task_signal_data import generate

    out_path = Path(model_path) if model_path else _default_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic data...", flush=True)
    data = generate(seed=seed)
    texts      = [row[0] for row in data]
    complexity = [row[1] for row in data]
    nature     = [row[2] for row in data]
    domain     = [row[3] for row in data]

    def _split(labels):
        return train_test_split(
            texts, labels, test_size=test_split, random_state=seed, stratify=labels
        )

    def _make_pipeline(C: float = 4.0) -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(1, 3),
                min_df=2,
                sublinear_tf=True,
                max_features=50_000,
            )),
            ("clf", LogisticRegression(
                C=C,
                max_iter=500,
                solver="lbfgs",
                random_state=seed,
            )),
        ])

    results = {}
    for head_name, labels in [("complexity", complexity), ("nature", nature), ("domain", domain)]:
        X_train, X_val, y_train, y_val = _split(labels)
        print(f"Training {head_name} head ({len(X_train)} train / {len(X_val)} val)...", flush=True)
        pipeline = _make_pipeline()
        pipeline.fit(X_train, y_train)
        acc = accuracy_score(y_val, pipeline.predict(X_val))
        print(f"  {head_name} accuracy: {acc:.3f}")
        if acc < min_accuracy:
            raise RuntimeError(
                f"{head_name} validation accuracy {acc:.3f} < required {min_accuracy:.3f}"
            )
        results[head_name] = pipeline

    joblib.dump(results, out_path)
    print(f"Model saved to {out_path}")


def _default_path() -> Path:
    return Path.home() / ".axor" / "models" / "task_signal.joblib"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TaskSignalClassifier")
    parser.add_argument("--out", type=Path, default=None, help="Output model path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    args = parser.parse_args()
    train(model_path=args.out, seed=args.seed, min_accuracy=args.min_accuracy)
