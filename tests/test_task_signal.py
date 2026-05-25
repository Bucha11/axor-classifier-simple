"""
Tests for TaskSignalClassifier.

Trains a model in a temp directory, then verifies label prediction
and confidence for known-good inputs.
"""
from __future__ import annotations

import time
import os
import pytest
from pathlib import Path

from axor_classifier_simple.task_signal import TaskSignalClassifier, ModelNotTrainedError
from axor_classifier_simple._model_security import UntrustedModelError
from axor_classifier_simple.train_task_signal import train


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("models") / "task_signal.joblib"
    train(model_path=path, seed=42)
    return path


def test_model_not_trained_raises():
    with pytest.raises(ModelNotTrainedError):
        TaskSignalClassifier(model_path="/nonexistent/path/model.joblib")


def test_group_writable_model_file_is_rejected(trained_model):
    if os.name == "nt":
        pytest.skip("POSIX mode-bit check")
    original_mode = trained_model.stat().st_mode
    try:
        trained_model.chmod(0o664)
        with pytest.raises(UntrustedModelError):
            TaskSignalClassifier(model_path=trained_model)
    finally:
        trained_model.chmod(original_mode)


@pytest.mark.asyncio
async def test_focused_mutative_coding(trained_model):
    clf = TaskSignalClassifier(model_path=trained_model)
    text = "fix the bug in parser.py"
    signal, confidence = await clf.classify(text)
    assert signal.complexity.value == "focused"
    assert signal.nature.value == "mutative"
    assert signal.domain == "coding"
    assert confidence > 0.7


@pytest.mark.asyncio
async def test_expansive_generative_research(trained_model):
    clf = TaskSignalClassifier(model_path=trained_model)
    text = (
        "Research the entire authentication landscape across all microservices "
        "and write a comprehensive security audit report with recommendations "
        "for every component, covering OAUTH, JWT, session management, and MFA"
    )
    signal, confidence = await clf.classify(text)
    assert signal.complexity.value == "expansive"
    assert signal.requires_children is True


@pytest.mark.asyncio
async def test_empty_input_does_not_raise(trained_model):
    clf = TaskSignalClassifier(model_path=trained_model)
    signal, confidence = await clf.classify("")
    assert signal is not None
    assert 0.0 <= confidence <= 1.0


@pytest.mark.asyncio
async def test_classify_with_scores_structure(trained_model):
    clf = TaskSignalClassifier(model_path=trained_model)
    signal, confidence, scores = await clf.classify_with_scores("write a hello world script")
    for prefix in ("complexity.", "nature.", "domain."):
        assert any(k.startswith(prefix) for k in scores), f"missing {prefix} scores"
    total_complexity = sum(v for k, v in scores.items() if k.startswith("complexity."))
    assert abs(total_complexity - 1.0) < 1e-4


@pytest.mark.asyncio
async def test_inference_time_is_interactive(trained_model):
    clf = TaskSignalClassifier(model_path=trained_model)
    text = "Add a cache layer to the database query function"
    # warmup
    for _ in range(10):
        await clf.classify(text)
    N = 200
    t0 = time.perf_counter()
    for _ in range(N):
        await clf.classify(text)
    elapsed_ms = (time.perf_counter() - t0) * 1000 / N
    # Keep this as a coarse regression guard. Sub-millisecond targets belong in
    # a benchmark suite because CI hosts and sandboxed runs are noisy.
    assert elapsed_ms < 10.0, f"inference took {elapsed_ms:.3f}ms (target < 10ms)"


@pytest.mark.asyncio
async def test_requires_mutation_for_mutative(trained_model):
    clf = TaskSignalClassifier(model_path=trained_model)
    signal, _ = await clf.classify("Fix the null pointer exception in the parser")
    if signal.nature.value == "mutative":
        assert signal.requires_mutation is True
