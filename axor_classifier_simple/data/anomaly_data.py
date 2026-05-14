"""
Synthetic data generator for MLAnomalyDetector.

Produces labelled sequences of NormalizedIntent windows with class labels
NORMAL / SUSPICIOUS / CRITICAL per spec distributions:
  70% NORMAL, 20% SUSPICIOUS, 10% CRITICAL  (~50k total)

Encoding-agnostic by design: encodes behavioral trajectories, not content.
"""
from __future__ import annotations

import random
from dataclasses import asdict

try:
    from axor_core.contracts.anomaly import NormalizedIntent
except ImportError:  # allow standalone use without axor-core installed
    NormalizedIntent = None  # type: ignore[assignment,misc]


# ── Intent factory helpers ─────────────────────────────────────────────────────

def _intent(
    *,
    tool: str = "read",
    operation: str = "file_read",
    target_kind: str = "workdir",
    destination_kind: str = "none",
    provenance: str = "repo",
    reads_secret: bool = False,
    writes_outside: bool = False,
    executes_generated: bool = False,
    after_external: bool = False,
    after_secret: bool = False,
    data_flow: str = "none",
) -> "NormalizedIntent":
    return NormalizedIntent(
        tool=tool,
        operation=operation,
        target_kind=target_kind,
        destination_kind=destination_kind,
        provenance=provenance,
        reads_secret_like_data=reads_secret,
        writes_outside_workdir=writes_outside,
        executes_generated_code=executes_generated,
        after_external_read=after_external,
        after_secret_access=after_secret,
        data_flow=data_flow,
    )


# ── Pattern builders ───────────────────────────────────────────────────────────

def _normal_coding_mutative(rng: random.Random) -> list["NormalizedIntent"]:
    patterns = [
        [_intent(operation="file_read"), _intent(operation="file_read"),
         _intent(tool="write", operation="file_write"), _intent(operation="test"),
         _intent(tool="write", operation="file_write")],
        [_intent(operation="file_read"), _intent(operation="search"),
         _intent(operation="file_read"), _intent(tool="write", operation="file_write")],
        [_intent(operation="search"), _intent(operation="file_read"),
         _intent(operation="file_read"), _intent(tool="write", operation="file_write"),
         _intent(operation="test")],
    ]
    seq = rng.choice(patterns)[:]
    _vary_normal(seq, rng)
    return seq


def _normal_coding_readonly(rng: random.Random) -> list["NormalizedIntent"]:
    seq = [_intent(operation="file_read") for _ in range(rng.randint(3, 6))]
    for i in range(len(seq)):
        if rng.random() < 0.3:
            seq[i] = _intent(operation="search")
    _vary_normal(seq, rng)
    return seq


def _normal_research_readonly(rng: random.Random) -> list["NormalizedIntent"]:
    seq = []
    for _ in range(rng.randint(4, 8)):
        if rng.random() < 0.5:
            seq.append(_intent(tool="web_search", operation="search"))
        else:
            seq.append(_intent(operation="file_read"))
    _vary_normal(seq, rng)
    return seq


def _normal_localhost_net(rng: random.Random) -> list["NormalizedIntent"]:
    # health check — legitimate local network call
    return [
        _intent(operation="file_read"),
        _intent(tool="bash", operation="network_request",
                target_kind="localhost", destination_kind="localhost",
                data_flow="local_to_local"),
        _intent(operation="file_read"),
    ]


def _vary_normal(seq: list, rng: random.Random) -> None:
    # inject benign variations
    if rng.random() < 0.3 and seq:
        idx = rng.randint(0, len(seq) - 1)
        seq.insert(idx, _intent(operation="search"))


def _suspicious_external_then_curl(rng: random.Random) -> list["NormalizedIntent"]:
    return [
        _intent(tool="web_fetch", operation="network_request",
                target_kind="external_url", provenance="external_web",
                data_flow="local_to_external"),
        _intent(operation="file_read"),
        _intent(tool="bash", operation="network_request",
                target_kind="external_url", destination_kind="external_domain",
                provenance="external_web", after_external=True,
                data_flow="local_to_external"),
    ]


def _suspicious_external_read_write_outside(rng: random.Random) -> list["NormalizedIntent"]:
    return [
        _intent(tool="web_fetch", operation="network_request",
                target_kind="external_url", provenance="external_web"),
        _intent(operation="file_read", target_kind="secret",
                reads_secret=True, provenance="external_web", after_external=True),
        _intent(tool="write", operation="file_write",
                target_kind="system_path", writes_outside=True,
                provenance="external_web", after_external=True),
    ]


def _suspicious_generated_code(rng: random.Random) -> list["NormalizedIntent"]:
    return [
        _intent(operation="file_read"),
        _intent(tool="write", operation="file_write"),
        _intent(tool="bash", operation="execute_generated_code",
                executes_generated=True, data_flow="local_to_local"),
    ]


def _critical_exfil(rng: random.Random) -> list["NormalizedIntent"]:
    return [
        _intent(tool="web_fetch", operation="network_request",
                target_kind="external_url", provenance="external_web",
                data_flow="local_to_external"),
        _intent(tool="read", operation="file_read",
                target_kind="secret", reads_secret=True,
                provenance="external_web", after_external=True),
        _intent(tool="bash", operation="network_request",
                target_kind="external_url", destination_kind="external_domain",
                provenance="external_web", after_external=True, after_secret=True,
                data_flow="local_to_external"),
    ]


def _critical_ssh_exfil(rng: random.Random) -> list["NormalizedIntent"]:
    return [
        _intent(tool="web_fetch", operation="network_request",
                target_kind="external_url", provenance="external_web"),
        _intent(tool="read", operation="file_read",
                target_kind="secret", reads_secret=True,
                provenance="external_web", after_external=True),
        _intent(tool="write", operation="file_write",
                target_kind="system_path", writes_outside=True,
                provenance="external_web", after_external=True, after_secret=True),
    ]


def _critical_cloud_metadata(rng: random.Random) -> list["NormalizedIntent"]:
    return [
        _intent(operation="file_read"),
        _intent(tool="bash", operation="network_request",
                target_kind="cloud_metadata", destination_kind="cloud_metadata",
                data_flow="local_to_external"),
    ]


def _critical_external_to_shell(rng: random.Random) -> list["NormalizedIntent"]:
    return [
        _intent(tool="web_fetch", operation="network_request",
                target_kind="external_url", provenance="external_web"),
        _intent(tool="write", operation="file_write", provenance="external_web",
                after_external=True),
        _intent(tool="bash", operation="execute_generated_code",
                executes_generated=True, after_external=True,
                data_flow="external_to_shell"),
    ]


# ── Padding ────────────────────────────────────────────────────────────────────

def _pad_window(seq: list, window_size: int, rng: random.Random) -> list:
    """Pad sequence to window_size with benign intents prepended."""
    while len(seq) < window_size:
        seq.insert(0, _intent(operation=rng.choice(["file_read", "search"])))
    return seq[-window_size:]


# ── Main generator ─────────────────────────────────────────────────────────────

_NORMAL_BUILDERS = [
    _normal_coding_mutative,
    _normal_coding_readonly,
    _normal_research_readonly,
    _normal_localhost_net,
]

_SUSPICIOUS_BUILDERS = [
    _suspicious_external_then_curl,
    _suspicious_external_read_write_outside,
    _suspicious_generated_code,
]

_CRITICAL_BUILDERS = [
    _critical_exfil,
    _critical_ssh_exfil,
    _critical_cloud_metadata,
    _critical_external_to_shell,
]


def generate(
    seed: int = 42,
    total: int = 51000,
    window_size: int = 10,
    normal_frac: float = 0.70,
    suspicious_frac: float = 0.20,
) -> list[tuple[list[dict], str]]:
    """
    Returns list of (window_dicts, label) tuples.

    window_dicts: list of dicts (NormalizedIntent as dict, for serialization)
    label: "normal" | "suspicious" | "critical"
    """
    if NormalizedIntent is None:
        raise ImportError("axor-core must be installed to generate anomaly data")

    rng = random.Random(seed)
    n_normal     = int(total * normal_frac)
    n_suspicious = int(total * suspicious_frac)
    n_critical   = total - n_normal - n_suspicious

    out: list[tuple[list[dict], str]] = []

    for _ in range(n_normal):
        builder = rng.choice(_NORMAL_BUILDERS)
        seq = builder(rng)
        window = _pad_window(seq, window_size, rng)
        out.append(([asdict(ni) for ni in window], "normal"))

    for _ in range(n_suspicious):
        builder = rng.choice(_SUSPICIOUS_BUILDERS)
        seq = builder(rng)
        window = _pad_window(seq, window_size, rng)
        out.append(([asdict(ni) for ni in window], "suspicious"))

    for _ in range(n_critical):
        builder = rng.choice(_CRITICAL_BUILDERS)
        seq = builder(rng)
        window = _pad_window(seq, window_size, rng)
        out.append(([asdict(ni) for ni in window], "critical"))

    rng.shuffle(out)
    return out
