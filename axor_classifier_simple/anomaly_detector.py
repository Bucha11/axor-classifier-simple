"""
MLAnomalyDetector — behavioral anomaly scoring for governed sessions.

Architecture:
  GradientBoostingClassifier on flattened NormalizedIntent window
  Observation window: configurable, default 10 intents
  Optional LLMVerifier for gray-zone cases

Implements AnomalyDetector Protocol from axor-core.
Inference target: < 1ms
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from axor_classifier_simple._model_security import validate_trusted_model_file

log = logging.getLogger("axor.anomaly_detector")

try:
    import joblib
    import numpy as np
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from axor_core.contracts.anomaly import (
        AnomalyClass,
        AnomalyDetector,
        AnomalyResult,
        LLMVerifier,
        NormalizedIntent,
    )
    from axor_core.node.canonicalizer import IntentCanonicalizer
    _AXOR_AVAILABLE = True
except ImportError:
    _AXOR_AVAILABLE = False

_DEFAULT_MODEL_PATH = Path(os.environ.get(
    "AXOR_ANOMALY_MODEL",
    Path.home() / ".axor" / "models" / "anomaly_detector.joblib",
))

# Feature field names in fixed order (must match train_anomaly.py)
_BOOLEAN_FIELDS = [
    "reads_secret_like_data",
    "writes_outside_workdir",
    "executes_generated_code",
    "after_external_read",
    "after_secret_access",
]

_OPERATION_VALUES = [
    "execute_generated_code", "file_read", "file_write",
    "network_request", "other", "package_install", "search", "test",
]

_TARGET_KIND_VALUES = [
    "cloud_metadata", "docker_socket", "external_url", "localhost",
    "private_network", "secret", "system_path", "workdir",
]

_DATA_FLOW_VALUES = [
    "external_to_shell", "local_to_external", "local_to_local", "none",
]

_PROVENANCE_VALUES = [
    "external_web", "official_docs", "repo", "unknown", "user",
]


def _require_sklearn() -> None:
    if not _SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn and joblib are required for MLAnomalyDetector. "
            "Install with: pip install axor-classifier-simple[ml]"
        )


def _intent_to_features(intent: "NormalizedIntent") -> list[float]:
    """Convert one NormalizedIntent to a fixed-length feature vector."""
    feats: list[float] = []

    # boolean flags
    for field in _BOOLEAN_FIELDS:
        feats.append(1.0 if getattr(intent, field) else 0.0)

    # one-hot: operation
    for val in _OPERATION_VALUES:
        feats.append(1.0 if intent.operation == val else 0.0)

    # one-hot: target_kind
    for val in _TARGET_KIND_VALUES:
        feats.append(1.0 if intent.target_kind == val else 0.0)

    # one-hot: data_flow
    for val in _DATA_FLOW_VALUES:
        feats.append(1.0 if intent.data_flow == val else 0.0)

    # one-hot: provenance
    for val in _PROVENANCE_VALUES:
        feats.append(1.0 if intent.provenance == val else 0.0)

    return feats


_FEATURES_PER_INTENT = (
    len(_BOOLEAN_FIELDS)
    + len(_OPERATION_VALUES)
    + len(_TARGET_KIND_VALUES)
    + len(_DATA_FLOW_VALUES)
    + len(_PROVENANCE_VALUES)
)


def window_to_feature_vector(
    window: "list[NormalizedIntent]",
    window_size: int = 10,
) -> list[float]:
    """
    Flatten a window of NormalizedIntents into a fixed-length feature vector.

    Pads with zeros if window is shorter than window_size.
    Truncates to last window_size if longer.
    """
    padded = window[-window_size:]
    feats: list[float] = []

    # pad on the left with zeros for missing intents
    for _ in range(window_size - len(padded)):
        feats.extend([0.0] * _FEATURES_PER_INTENT)

    for intent in padded:
        feats.extend(_intent_to_features(intent))

    return feats


def _score_to_class(
    score: float,
    suspicious_threshold: float = 0.40,
    critical_threshold: float = 0.75,
) -> "AnomalyClass":
    if score >= critical_threshold:
        return AnomalyClass.CRITICAL
    if score >= suspicious_threshold:
        return AnomalyClass.SUSPICIOUS
    return AnomalyClass.NORMAL


def _extract_reasons(
    window: "list[NormalizedIntent]",
    cls: "AnomalyClass",
) -> list[str]:
    """Derive human-readable trigger reasons from the window."""
    reasons: list[str] = []
    last = window[-1] if window else None

    external_seen = any(i.after_external_read or i.target_kind == "external_url" for i in window)
    secret_seen   = any(i.after_secret_access or i.reads_secret_like_data for i in window)

    if external_seen:
        reasons.append("external_read_seen")
    if secret_seen and external_seen:
        reasons.append("secret_access_after_external_read")
    if last and last.data_flow in ("local_to_external", "external_to_shell") and secret_seen:
        reasons.append("network_activity_after_secret_access")
    if any(i.executes_generated_code for i in window):
        reasons.append("executes_generated_code")
    if any(i.writes_outside_workdir for i in window):
        reasons.append("writes_outside_workdir")
    if any(i.target_kind == "cloud_metadata" for i in window):
        reasons.append("cloud_metadata_access")
    if any(i.target_kind == "docker_socket" for i in window):
        reasons.append("docker_socket_access")
    if last and last.data_flow == "external_to_shell":
        reasons.append("external_to_shell_execution")

    return reasons or [cls.value]


if _AXOR_AVAILABLE:
    class MLAnomalyDetector:
        """
        GradientBoostingClassifier-based AnomalyDetector.

        Implements the AnomalyDetector Protocol from axor-core structurally.
        Optional gray_zone_verifier (LLMVerifier) is called for SUSPICIOUS scores
        where the model is uncertain.

        gray_zone_verifier: called when score in [0.40, 0.75) — the gray zone.
        gray_zone_threshold: minimum confidence required before calling verifier.
        """

        def __init__(
            self,
            model_path: Path | str | None = None,
            gray_zone_verifier: "LLMVerifier | None" = None,
            window_size: int = 10,
            gray_zone_threshold: float = 0.50,
            suspicious_threshold: float = 0.40,
            critical_threshold: float = 0.75,
            score_weights: "dict[str, float] | None" = None,
            fail_closed_on_verifier_error: bool = False,
        ) -> None:
            _require_sklearn()
            path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
            if not path.exists():
                raise ModelNotTrainedError(
                    f"No trained anomaly model found at {path}. "
                    "Run: python -m axor_classifier_simple.train_anomaly"
                )
            self._model                = joblib.load(validate_trusted_model_file(path))
            self._verifier             = gray_zone_verifier
            self._window_size          = window_size
            self._gray_zone_threshold  = gray_zone_threshold
            self._suspicious_threshold = suspicious_threshold
            self._critical_threshold   = critical_threshold
            self._score_weights        = score_weights or {
                "critical": 1.0, "suspicious": 0.55, "normal": 0.0
            }
            self._canonicalizer        = IntentCanonicalizer()
            self._fail_closed_on_verifier_error = fail_closed_on_verifier_error

        async def score(
            self,
            window: "list[NormalizedIntent]",
            task_signal_hint: str = "",
            policy_name: str = "",
        ) -> "AnomalyResult":
            if not window:
                return AnomalyResult(score=0.0, cls=AnomalyClass.NORMAL)

            fv = window_to_feature_vector(window, self._window_size)
            proba = self._model.predict_proba([fv])[0]

            # model classes are sorted alphabetically: critical, normal, suspicious
            classes = list(self._model.classes_)
            score = _weighted_score(proba, classes, self._score_weights)
            cls   = _score_to_class(score, self._suspicious_threshold, self._critical_threshold)

            # gray zone: optionally escalate to LLM verifier
            if (
                cls == AnomalyClass.SUSPICIOUS
                and self._verifier is not None
                and score >= self._gray_zone_threshold
            ):
                try:
                    canonical_window = [
                        self._canonicalizer.canonicalize(intent) for intent in window
                    ]
                    result = await self._verifier.verify(
                        window=canonical_window,
                        task_signal_hint=task_signal_hint,
                        policy_name=policy_name,
                    )
                    return result
                except Exception as exc:
                    log.warning("LLM verifier failed: %s", exc)
                    if self._fail_closed_on_verifier_error:
                        return AnomalyResult(
                            score=min(1.0, max(score, self._critical_threshold)),
                            cls=AnomalyClass.CRITICAL,
                            reasons=("verifier_error_fail_closed",),
                        )

            reasons = _extract_reasons(window, cls)
            return AnomalyResult(score=score, cls=cls, reasons=tuple(reasons))

else:
    class MLAnomalyDetector:  # type: ignore[no-redef]
        """Stub when axor-core is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "axor-core is required for MLAnomalyDetector. "
                "Install with: pip install axor-core axor-classifier-simple[ml]"
            )


class ModelNotTrainedError(RuntimeError):
    """Raised when the anomaly model has not been trained yet."""


# ── Score helpers ──────────────────────────────────────────────────────────────

def _weighted_score(
    proba: "np.ndarray",
    classes: list[str],
    weights: "dict[str, float]",
) -> float:
    """Convert class probabilities to a 0.0–1.0 risk score using caller-supplied weights."""
    score = 0.0
    for cls_name, p in zip(classes, proba):
        score += weights.get(cls_name, 0.0) * float(p)
    return min(1.0, score)
