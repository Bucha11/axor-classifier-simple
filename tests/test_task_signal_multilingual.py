"""
Multilingual and out-of-distribution behaviour of TaskSignalClassifier.

Russian is a first-class training language (the _TEMPLATES_RU corpus):
clear Russian tasks must classify as accurately as English ones. OOD
garbage must never crash and must not come back with near-certain
confidence — in the axor-core escalation cascade an over-confident wrong
answer silently overrides the heuristic.
"""
from __future__ import annotations

import pytest

from axor_classifier_simple.task_signal import TaskSignalClassifier
from axor_classifier_simple.train_task_signal import train


# ── Russian classification ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ru_focused_mutative_coding(trained_model):
    clf = TaskSignalClassifier(model_path=trained_model)
    signal, confidence = await clf.classify("исправь баг в auth.py")
    assert signal.complexity.value == "focused"
    assert signal.nature.value == "mutative"
    assert signal.domain == "coding"
    assert confidence > 0.7


@pytest.mark.asyncio
async def test_ru_focused_readonly(trained_model):
    clf = TaskSignalClassifier(model_path=trained_model)
    signal, confidence = await clf.classify("объясни, что делает validate")
    assert signal.complexity.value == "focused"
    assert signal.nature.value == "readonly"
    assert confidence > 0.7


@pytest.mark.asyncio
async def test_ru_expansive_mutative(trained_model):
    clf = TaskSignalClassifier(model_path=trained_model)
    signal, _ = await clf.classify("перепиши весь репозиторий с нуля")
    assert signal.complexity.value == "expansive"
    assert signal.nature.value == "mutative"
    assert signal.requires_children is True
    assert signal.requires_mutation is True


@pytest.mark.asyncio
async def test_ru_focused_generative(trained_model):
    clf = TaskSignalClassifier(model_path=trained_model)
    signal, _ = await clf.classify("напиши тест для authenticate")
    assert signal.complexity.value == "focused"
    assert signal.nature.value == "generative"


@pytest.mark.asyncio
async def test_ru_support_domain(trained_model):
    clf = TaskSignalClassifier(model_path=trained_model)
    signal, _ = await clf.classify("почему redis возвращает TimeoutError")
    assert signal.nature.value == "readonly"
    assert signal.domain == "support"


# ── Out-of-distribution robustness ───────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "garbage",
    [
        "🙂🙂🙂 42 ### @@@",
        "....",
        "™©®",
        "zzz zzz zzz",
        "0101010101 0101010101",
    ],
)
async def test_garbage_input_does_not_raise_and_is_not_overconfident(
    trained_model, garbage
):
    clf = TaskSignalClassifier(model_path=trained_model)
    signal, confidence = await clf.classify(garbage)
    assert signal is not None
    assert 0.0 <= confidence <= 1.0
    # A signal-free input must not out-shout the heuristic with near-certainty.
    assert confidence < 0.9, f"overconfident on garbage: {garbage!r} -> {confidence}"


@pytest.mark.asyncio
async def test_unseen_script_does_not_raise(trained_model):
    """A script absent from training (Chinese) must degrade, not crash."""
    clf = TaskSignalClassifier(model_path=trained_model)
    signal, confidence = await clf.classify("修复解析器中的错误")
    assert signal is not None
    assert 0.0 <= confidence <= 1.0


# ── Training metrics contract ────────────────────────────────────────────────────

def test_train_reports_calibration(tmp_path):
    results = train(model_path=tmp_path / "m.joblib", seed=7)
    for head in ("complexity", "nature", "domain"):
        assert 0.0 <= results[head]["hard_ece"] <= 1.0
        assert results[head]["hard_acc"] >= 0.75
