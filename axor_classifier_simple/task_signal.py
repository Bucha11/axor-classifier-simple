"""
TaskSignalClassifier — ML replacement for HeuristicClassifier.

Architecture:
  TF-IDF (char n-grams 1-3 + word n-grams 1-3)
  Three independent LogisticRegression heads: complexity, nature, domain
  Serialized with joblib

Inference target: < 1ms
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    import joblib
    import numpy as np
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from axor_core.contracts.policy import (
        SignalClassifier,
        TaskSignal,
        TaskComplexity,
        TaskNature,
    )
    _AXOR_AVAILABLE = True
except ImportError:
    _AXOR_AVAILABLE = False

_DEFAULT_MODEL_PATH = Path(os.environ.get(
    "AXOR_TASK_SIGNAL_MODEL",
    Path.home() / ".axor" / "models" / "task_signal.joblib",
))

_COMPLEXITY_CLASSES = ["focused", "moderate", "expansive"]
_NATURE_CLASSES = ["generative", "mutative", "readonly"]
_DOMAIN_CLASSES = ["analysis", "coding", "general", "research", "support"]

_COMPLEXITY_SCOPE = {"focused": 1, "moderate": 5, "expansive": 999}


class ModelNotTrainedError(RuntimeError):
    """Raised when the model file has not been trained yet."""


def _require_sklearn() -> None:
    if not _SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn and joblib are required for TaskSignalClassifier. "
            "Install with: pip install axor-classifier-simple[ml]"
        )


if _AXOR_AVAILABLE:
    class TaskSignalClassifier(SignalClassifier):
        """
        ML-based TaskSignal classifier implementing the SignalClassifier ABC.

        Three independent TF-IDF + LogisticRegression heads: complexity, nature, domain.
        Load a trained model with TaskSignalClassifier(model_path=...).
        Train with axor_classifier_simple.train_task_signal.train().
        """

        def __init__(self, model_path: Path | str | None = None) -> None:
            _require_sklearn()
            path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
            if not path.exists():
                raise ModelNotTrainedError(
                    f"No trained model found at {path}. "
                    "Run: python -m axor_classifier_simple.train_task_signal"
                )
            bundle = joblib.load(path)
            self._complexity_pipeline = bundle["complexity"]
            self._nature_pipeline     = bundle["nature"]
            self._domain_pipeline     = bundle["domain"]

        async def classify(self, raw_input: str) -> tuple[TaskSignal, float]:
            signal, confidence, _ = await self.classify_with_scores(raw_input)
            return signal, confidence

        async def classify_with_scores(
            self, raw_input: str
        ) -> tuple[TaskSignal, float, dict[str, float]]:
            text = raw_input.strip()
            if not text:
                text = "unknown"

            c_proba = self._complexity_pipeline.predict_proba([text])[0]
            n_proba = self._nature_pipeline.predict_proba([text])[0]
            d_proba = self._domain_pipeline.predict_proba([text])[0]

            c_idx = int(np.argmax(c_proba))
            n_idx = int(np.argmax(n_proba))
            d_idx = int(np.argmax(d_proba))

            complexity_str = str(self._complexity_pipeline.classes_[c_idx])
            nature_str = str(self._nature_pipeline.classes_[n_idx])
            domain_str = str(self._domain_pipeline.classes_[d_idx])

            complexity = TaskComplexity(complexity_str)
            nature = TaskNature(nature_str)
            confidence = float(min(c_proba[c_idx], n_proba[n_idx]))

            signal = TaskSignal(
                raw_input=raw_input,
                complexity=complexity,
                nature=nature,
                estimated_scope=_scope(complexity),
                requires_children=complexity == TaskComplexity.EXPANSIVE,
                requires_mutation=nature == TaskNature.MUTATIVE,
                domain=domain_str,
            )

            scores: dict[str, float] = {}
            for label, prob in zip(self._complexity_pipeline.classes_, c_proba):
                scores[f"complexity.{label}"] = float(prob)
            for label, prob in zip(self._nature_pipeline.classes_, n_proba):
                scores[f"nature.{label}"] = float(prob)
            for label, prob in zip(self._domain_pipeline.classes_, d_proba):
                scores[f"domain.{label}"] = float(prob)

            return signal, confidence, scores

else:
    class TaskSignalClassifier:  # type: ignore[no-redef]
        """Stub when axor-core is not installed."""

        def __init__(self, model_path=None):
            raise ImportError(
                "axor-core is required for TaskSignalClassifier. "
                "Install with: pip install axor-core axor-classifier-simple[ml]"
            )


def _scope(complexity: "TaskComplexity") -> int:
    return _COMPLEXITY_SCOPE[complexity.value]
