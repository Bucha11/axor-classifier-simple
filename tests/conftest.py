from __future__ import annotations

from pathlib import Path

import pytest

from axor_classifier_simple.train_task_signal import train


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory) -> Path:
    """Train the task-signal model once per test session — training the full
    EN+RU corpus is the expensive part, every test module shares the result."""
    path = tmp_path_factory.mktemp("models") / "task_signal.joblib"
    train(model_path=path, seed=42)
    return path
