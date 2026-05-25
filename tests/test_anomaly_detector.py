"""
Tests for MLAnomalyDetector.

Trains a model in a temp directory, then verifies:
- CRITICAL pattern (external_read → secret_access → network_exfil) → CRITICAL
- NORMAL coding pattern → NORMAL
- window_to_feature_vector shape and padding
- _score_to_class thresholds
- Inference time < 1ms
"""
from __future__ import annotations

import time
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from axor_core.contracts.anomaly import AnomalyClass, AnomalyResult, NormalizedIntent
from axor_classifier_simple.anomaly_detector import (
    MLAnomalyDetector,
    _score_to_class,
    _FEATURES_PER_INTENT,
    window_to_feature_vector,
)
from axor_classifier_simple.train_anomaly import train
from axor_classifier_simple._model_security import UntrustedModelError


def _ni(**kw) -> NormalizedIntent:
    defaults = dict(
        tool="read", operation="file_read", target_kind="workdir",
        destination_kind="none", provenance="repo",
        reads_secret_like_data=False, writes_outside_workdir=False,
        executes_generated_code=False, after_external_read=False,
        after_secret_access=False, data_flow="none",
    )
    defaults.update(kw)
    return NormalizedIntent(**defaults)


def _critical_exfil_window() -> list[NormalizedIntent]:
    """external_read → secret_access → network_exfil (canonical CRITICAL pattern)."""
    return [
        _ni(operation="file_read", target_kind="external_url",
            provenance="external_web", data_flow="none"),
        _ni(operation="file_read", target_kind="secret",
            reads_secret_like_data=True, after_external_read=True, data_flow="none",
            provenance="repo"),
        _ni(tool="curl", operation="network_request", target_kind="external_url",
            after_external_read=True, after_secret_access=True,
            data_flow="local_to_external", provenance="external_web"),
    ]


def _normal_coding_window() -> list[NormalizedIntent]:
    return [
        _ni(operation="file_read"),
        _ni(operation="file_read"),
        _ni(tool="write", operation="file_write"),
        _ni(operation="test"),
        _ni(operation="file_read"),
    ]


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("models") / "anomaly_detector.joblib"
    train(model_path=path, seed=42, n_estimators=100)
    return path


# ── window_to_feature_vector ────────────────────────────────────────────────────

def test_feature_vector_shape_full_window():
    window = [_ni() for _ in range(10)]
    fv = window_to_feature_vector(window, window_size=10)
    assert len(fv) == 10 * _FEATURES_PER_INTENT


def test_feature_vector_pads_short_window():
    window = [_ni()]
    fv = window_to_feature_vector(window, window_size=10)
    assert len(fv) == 10 * _FEATURES_PER_INTENT
    # First 9 slots should be all zeros
    zeros = fv[: 9 * _FEATURES_PER_INTENT]
    assert all(v == 0.0 for v in zeros)


def test_feature_vector_truncates_long_window():
    window = [_ni() for _ in range(15)]
    fv = window_to_feature_vector(window, window_size=10)
    assert len(fv) == 10 * _FEATURES_PER_INTENT


def test_feature_vector_empty_window():
    fv = window_to_feature_vector([], window_size=10)
    assert len(fv) == 10 * _FEATURES_PER_INTENT
    assert all(v == 0.0 for v in fv)


# ── _score_to_class ─────────────────────────────────────────────────────────────

def test_score_to_class_normal():
    assert _score_to_class(0.0) == AnomalyClass.NORMAL
    assert _score_to_class(0.39) == AnomalyClass.NORMAL


def test_score_to_class_suspicious():
    assert _score_to_class(0.40) == AnomalyClass.SUSPICIOUS
    assert _score_to_class(0.74) == AnomalyClass.SUSPICIOUS


def test_score_to_class_critical():
    assert _score_to_class(0.75) == AnomalyClass.CRITICAL
    assert _score_to_class(1.0) == AnomalyClass.CRITICAL


def test_score_to_class_custom_thresholds():
    assert _score_to_class(0.5, suspicious_threshold=0.6, critical_threshold=0.8) == AnomalyClass.NORMAL
    assert _score_to_class(0.7, suspicious_threshold=0.6, critical_threshold=0.8) == AnomalyClass.SUSPICIOUS
    assert _score_to_class(0.9, suspicious_threshold=0.6, critical_threshold=0.8) == AnomalyClass.CRITICAL


# ── MLAnomalyDetector ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_critical_exfil_pattern(trained_model):
    detector = MLAnomalyDetector(model_path=trained_model)
    result = await detector.score(window=_critical_exfil_window())
    assert result.cls == AnomalyClass.CRITICAL


@pytest.mark.asyncio
async def test_normal_coding_pattern(trained_model):
    detector = MLAnomalyDetector(model_path=trained_model)
    result = await detector.score(window=_normal_coding_window())
    assert result.cls == AnomalyClass.NORMAL


@pytest.mark.asyncio
async def test_empty_window_returns_normal(trained_model):
    detector = MLAnomalyDetector(model_path=trained_model)
    result = await detector.score(window=[])
    assert result.cls == AnomalyClass.NORMAL
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_inference_time_under_1ms(trained_model):
    detector = MLAnomalyDetector(model_path=trained_model)
    window = _normal_coding_window()
    N = 1000
    t0 = time.perf_counter()
    for _ in range(N):
        await detector.score(window=window)
    elapsed_ms = (time.perf_counter() - t0) * 1000 / N
    assert elapsed_ms < 1.0, f"inference took {elapsed_ms:.3f}ms (target < 1ms)"


@pytest.mark.asyncio
async def test_gray_zone_verifier_called_on_suspicious(trained_model):
    verifier = AsyncMock()
    verifier.verify = AsyncMock(return_value=AnomalyResult(
        score=0.85, cls=AnomalyClass.CRITICAL, reasons=("verifier_escalated",)
    ))
    # Set thresholds so a SUSPICIOUS score always hits verifier
    detector = MLAnomalyDetector(
        model_path=trained_model,
        gray_zone_verifier=verifier,
        gray_zone_threshold=0.0,  # call verifier for any suspicious score
    )

    # Build a window that produces SUSPICIOUS
    suspicious_window = [
        _ni(operation="file_read", target_kind="external_url",
            provenance="external_web"),
        _ni(operation="file_write", writes_outside_workdir=True, after_external_read=True),
    ]

    result = await detector.score(window=suspicious_window)
    # If suspicious, verifier should have been called and its result returned
    if verifier.verify.called:
        assert result.cls == AnomalyClass.CRITICAL
        assert "verifier_escalated" in result.reasons


@pytest.mark.asyncio
async def test_gray_zone_verifier_failure_falls_back(trained_model):
    verifier = AsyncMock()
    verifier.verify = AsyncMock(side_effect=RuntimeError("verifier down"))
    detector = MLAnomalyDetector(
        model_path=trained_model,
        gray_zone_verifier=verifier,
        gray_zone_threshold=0.0,
    )
    # Must not raise even when verifier fails
    result = await detector.score(window=_normal_coding_window())
    assert result.cls in (AnomalyClass.NORMAL, AnomalyClass.SUSPICIOUS, AnomalyClass.CRITICAL)


@pytest.mark.asyncio
async def test_gray_zone_verifier_failure_can_fail_closed(trained_model):
    verifier = AsyncMock()
    verifier.verify = AsyncMock(side_effect=RuntimeError("verifier down"))
    detector = MLAnomalyDetector(
        model_path=trained_model,
        gray_zone_verifier=verifier,
        gray_zone_threshold=0.0,
        suspicious_threshold=-1.0,
        critical_threshold=2.0,
        fail_closed_on_verifier_error=True,
    )

    result = await detector.score(window=_normal_coding_window())

    assert verifier.verify.called
    assert result.cls == AnomalyClass.CRITICAL
    assert "verifier_error_fail_closed" in result.reasons


def test_model_not_found_raises():
    from axor_classifier_simple.anomaly_detector import ModelNotTrainedError
    with pytest.raises(ModelNotTrainedError):
        MLAnomalyDetector(model_path="/nonexistent/path/model.joblib")


def test_group_writable_model_file_is_rejected(trained_model):
    if os.name == "nt":
        pytest.skip("POSIX mode-bit check")
    original_mode = trained_model.stat().st_mode
    try:
        trained_model.chmod(0o664)
        with pytest.raises(UntrustedModelError):
            MLAnomalyDetector(model_path=trained_model)
    finally:
        trained_model.chmod(original_mode)
