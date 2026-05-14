# axor-classifier-simple

[![PyPI](https://img.shields.io/pypi/v/axor-classifier-simple)](https://pypi.org/project/axor-classifier-simple/)
[![Python](https://img.shields.io/pypi/pyversions/axor-classifier-simple)](https://pypi.org/project/axor-classifier-simple/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ML classifiers for axor-core: task signal classification and behavioral anomaly detection.**

Two independent components, zero required dependencies — scikit-learn is an optional extra.

---

## What's included

| Component | Model | Inference target |
|-----------|-------|------------------|
| `TaskSignalClassifier` | TF-IDF + LogisticRegression (3 independent heads) | < 1 ms |
| `MLAnomalyDetector` | GradientBoostingClassifier + optional LLM verifier | < 1 ms |

Both implement protocols from `axor-core` and plug in with zero coupling to core internals.

---

## Installation

```bash
pip install axor-core axor-classifier-simple[ml]
```

Without `[ml]`, the package installs with no dependencies but raises `ImportError` on instantiation with an actionable message.

---

## TaskSignalClassifier

ML replacement for the built-in `HeuristicClassifier`. Implements the `SignalClassifier` ABC from `axor_core.contracts.policy`.

### Architecture

Three independent TF-IDF (char n-grams 1–3, `char_wb` analyzer) + LogisticRegression heads:

| Head | Labels |
|------|--------|
| `complexity` | `focused` · `moderate` · `expansive` |
| `nature` | `generative` · `mutative` · `readonly` |
| `domain` | `analysis` · `coding` · `general` · `research` · `support` |

Confidence is reported as `min(complexity_confidence, nature_confidence)`. Domain is a hint used by `GovernedSession` for policy defaults.

### Train

```bash
# train and save to ~/.axor/models/task_signal.joblib
python -m axor_classifier_simple.train_task_signal

# custom output path
python -m axor_classifier_simple.train_task_signal --out /path/to/model.joblib

# fail if validation accuracy < threshold (default 0.85)
python -m axor_classifier_simple.train_task_signal --min-accuracy 0.90
```

Training generates synthetic data, trains three pipelines, validates each head, and saves a joblib bundle. Raises `RuntimeError` if any head falls below `min_accuracy`.

### Use with axor-core

```python
from axor_classifier_simple import TaskSignalClassifier
from axor_core import GovernedSession, CapabilityExecutor

classifier = TaskSignalClassifier()  # loads ~/.axor/models/task_signal.joblib

session = GovernedSession(
    executor=my_executor,
    capability_executor=cap_executor,
    classifier=classifier,
    # heuristic runs first; escalates to ML only when confidence < 0.75
)
result = await session.run("refactor the auth module")
```

Custom model path or environment variable:

```bash
export AXOR_TASK_SIGNAL_MODEL=/path/to/task_signal.joblib
```

```python
classifier = TaskSignalClassifier(model_path="/path/to/task_signal.joblib")
```

### Inspect raw scores

```python
signal, confidence, scores = await classifier.classify_with_scores("write a test for /login")
# signal.complexity → TaskComplexity.FOCUSED
# signal.nature     → TaskNature.GENERATIVE
# confidence        → 0.91
# scores            → {"complexity.focused": 0.91, "nature.generative": 0.88, "domain.coding": 0.76, ...}
```

---

## MLAnomalyDetector

`GradientBoostingClassifier` that scores behavioral trajectories from sequences of `NormalizedIntent` objects. Implements the `AnomalyDetector` Protocol from `axor_core.contracts.anomaly`.

Optionally delegates gray-zone cases to an `LLMVerifier` (e.g. `LLMAnomalyVerifier` from `axor-classifier-llm`).

### Score thresholds

| Class | Score range |
|-------|-------------|
| `NORMAL` | `[0.0, 0.40)` |
| `SUSPICIOUS` | `[0.40, 0.75)` |
| `CRITICAL` | `[0.75, 1.0]` |

### Train

```bash
# train and save to ~/.axor/models/anomaly_detector.joblib
python -m axor_classifier_simple.train_anomaly

# custom options
python -m axor_classifier_simple.train_anomaly \
    --out /path/to/model.joblib \
    --n-estimators 300 \
    --max-depth 5 \
    --min-accuracy 0.92
```

Raises `RuntimeError` if validation accuracy falls below `min_accuracy` (default 0.90).

### Basic use

```python
from axor_classifier_simple import MLAnomalyDetector

detector = MLAnomalyDetector()  # loads ~/.axor/models/anomaly_detector.joblib

result = await detector.score(
    window=normalized_intents,
    task_signal_hint="focused_mutative",
    policy_name="focused_mutative",
)
print(result.score)    # float 0.0 – 1.0
print(result.cls)      # AnomalyClass.NORMAL / SUSPICIOUS / CRITICAL
print(result.reasons)  # ("external_read_seen", "executes_generated_code", ...)
```

Custom model path:

```bash
export AXOR_ANOMALY_MODEL=/path/to/anomaly_detector.joblib
```

```python
detector = MLAnomalyDetector(model_path="/path/to/anomaly_detector.joblib")
```

### With LLM verifier for gray-zone escalation

```python
import anthropic
from axor_classifier_simple import MLAnomalyDetector
from axor_classifier_llm import LLMAnomalyVerifier

verifier = LLMAnomalyVerifier(client=anthropic.AsyncAnthropic())

detector = MLAnomalyDetector(
    gray_zone_verifier=verifier,
    gray_zone_threshold=0.50,   # call verifier when score >= 0.50 and class is SUSPICIOUS
)
```

If the LLM call fails, the detector falls back to the ML score with a warning log.

### Constructor parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | `~/.axor/models/anomaly_detector.joblib` | Path to trained model |
| `gray_zone_verifier` | `None` | `LLMVerifier` for uncertain cases |
| `window_size` | `10` | Number of intents per observation window |
| `gray_zone_threshold` | `0.50` | Min score to invoke verifier in suspicious range |
| `suspicious_threshold` | `0.40` | Score boundary NORMAL / SUSPICIOUS |
| `critical_threshold` | `0.75` | Score boundary SUSPICIOUS / CRITICAL |
| `score_weights` | `{"critical": 1.0, "suspicious": 0.55, "normal": 0.0}` | Class probability weights |

---

## Feature encoding

Each `NormalizedIntent` is encoded to a fixed-length feature vector. Windows of 10 intents produce **300 features** total (shorter windows are zero-padded on the left).

| Group | Fields | Size |
|-------|--------|------|
| Boolean flags | `reads_secret_like_data`, `writes_outside_workdir`, `executes_generated_code`, `after_external_read`, `after_secret_access` | 5 |
| `operation` | `execute_generated_code`, `file_read`, `file_write`, `network_request`, `other`, `package_install`, `search`, `test` | 8 (one-hot) |
| `target_kind` | `cloud_metadata`, `docker_socket`, `external_url`, `localhost`, `private_network`, `secret`, `system_path`, `workdir` | 8 (one-hot) |
| `data_flow` | `external_to_shell`, `local_to_external`, `local_to_local`, `none` | 4 (one-hot) |
| `provenance` | `external_web`, `official_docs`, `repo`, `unknown`, `user` | 5 (one-hot) |

Feature order is fixed — it must match between training and inference. The encoding is defined in `anomaly_detector.py` and `data/anomaly_data.py`.

---

## Development

```bash
git clone https://github.com/Bucha11/axor-classifier-simple
cd axor-classifier-simple
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT
