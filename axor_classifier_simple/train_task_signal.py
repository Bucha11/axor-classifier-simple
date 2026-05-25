"""
Train and serialize the TaskSignalClassifier model.

Usage:
    python -m axor_classifier_simple.train_task_signal
    python -m axor_classifier_simple.train_task_signal --out /path/to/model.joblib

Two accuracy metrics are reported:
    synthetic val  — held-out slice of the template-generated corpus.
                     Expected: ~99%+. High accuracy here is easy; don't over-index on it.
    hard eval      — separate set of phrases not derived from any training template.
                     This is the real quality signal. Target: 80%+.

Feature extraction:
    FeatureUnion of two TF-IDF vectorizers:
      char_wb (1-4 n-grams): morphological patterns — "refactor" ≠ "factor"
      word    (1-2 n-grams): phrasal patterns — "entire codebase", "unit test",
                             scope markers like "just", "all", "every"
"""
from __future__ import annotations

import argparse
from pathlib import Path


def train(
    model_path: Path | str | None = None,
    seed: int = 42,
    test_split: float = 0.15,
    min_synthetic_accuracy: float = 0.90,
    min_hard_accuracy: float = 0.75,
) -> dict[str, dict[str, float]]:
    """
    Generate training data, train all three heads, evaluate, and save.

    Returns a dict of {head: {synthetic_acc, hard_acc}}.
    Raises RuntimeError if either accuracy gate is not met.
    """
    try:
        import joblib
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline, FeatureUnion
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
    except ImportError as e:
        raise ImportError(
            "scikit-learn and joblib are required. "
            "Install with: pip install axor-classifier-simple[ml]"
        ) from e

    from axor_classifier_simple.data.task_signal_data import generate, generate_hard

    out_path = Path(model_path) if model_path else _default_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Training corpus ────────────────────────────────────────────────────────
    print("Generating synthetic training data...", flush=True)
    data = generate(seed=seed)
    texts      = [row[0] for row in data]
    complexity = [row[1] for row in data]
    nature     = [row[2] for row in data]
    domain     = [row[3] for row in data]
    print(f"  {len(texts)} training examples across {len(set(zip(complexity, nature, domain)))} label combos")

    # ── Hard eval corpus ───────────────────────────────────────────────────────
    print("Loading hard eval set...", flush=True)
    hard = generate_hard(seed=seed)
    hard_texts      = [row[0] for row in hard]
    hard_complexity = [row[1] for row in hard]
    hard_nature     = [row[2] for row in hard]
    hard_domain     = [row[3] for row in hard]
    print(f"  {len(hard_texts)} hard eval examples (not used for training)")

    def _split(labels):
        return train_test_split(
            texts, labels, test_size=test_split, random_state=seed, stratify=labels
        )

    def _make_pipeline() -> Pipeline:
        features = FeatureUnion([
            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(1, 4),      # +4-gram vs baseline; "entire" as unit
                min_df=2,
                sublinear_tf=True,
                max_features=40_000,
            )),
            ("word", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),      # "entire codebase", "unit test", "just fix"
                min_df=2,
                sublinear_tf=True,
                max_features=20_000,
            )),
        ])
        return Pipeline([
            ("features", features),
            ("clf", LogisticRegression(
                C=4.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=seed,
            )),
        ])

    results = {}
    head_labels = [
        ("complexity", complexity, hard_complexity),
        ("nature",     nature,     hard_nature),
        ("domain",     domain,     hard_domain),
    ]

    for head_name, labels, hard_labels in head_labels:
        X_train, X_val, y_train, y_val = _split(labels)
        print(f"\nTraining '{head_name}' head ({len(X_train)} train / {len(X_val)} val)...", flush=True)

        pipeline = _make_pipeline()
        pipeline.fit(X_train, y_train)

        syn_acc  = accuracy_score(y_val,      pipeline.predict(X_val))
        hard_acc = accuracy_score(hard_labels, pipeline.predict(hard_texts))

        print(f"  synthetic val : {syn_acc:.3f}")
        print(f"  hard eval     : {hard_acc:.3f}  ← real quality signal")

        if syn_acc < min_synthetic_accuracy:
            raise RuntimeError(
                f"'{head_name}' synthetic val accuracy {syn_acc:.3f} "
                f"< required {min_synthetic_accuracy:.3f}"
            )
        if hard_acc < min_hard_accuracy:
            raise RuntimeError(
                f"'{head_name}' hard eval accuracy {hard_acc:.3f} "
                f"< required {min_hard_accuracy:.3f}"
            )

        results[head_name] = {
            "synthetic_acc": syn_acc,
            "hard_acc":      hard_acc,
            "pipeline":      pipeline,
        }

    bundle = {k: v["pipeline"] for k, v in results.items()}
    joblib.dump(bundle, out_path)
    print(f"\nModel saved to {out_path}")

    return {k: {"synthetic_acc": v["synthetic_acc"], "hard_acc": v["hard_acc"]}
            for k, v in results.items()}


def _default_path() -> Path:
    return Path.home() / ".axor" / "models" / "task_signal.joblib"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TaskSignalClassifier")
    parser.add_argument("--out",  type=Path,  default=None)
    parser.add_argument("--seed", type=int,   default=42)
    parser.add_argument("--min-synthetic-accuracy", type=float, default=0.90)
    parser.add_argument("--min-hard-accuracy",      type=float, default=0.75)
    args = parser.parse_args()
    train(
        model_path=args.out,
        seed=args.seed,
        min_synthetic_accuracy=args.min_synthetic_accuracy,
        min_hard_accuracy=args.min_hard_accuracy,
    )
