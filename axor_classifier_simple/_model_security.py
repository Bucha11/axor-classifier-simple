from __future__ import annotations

import stat
from pathlib import Path


class UntrustedModelError(RuntimeError):
    """Raised when a joblib model file is writable by non-owner users."""


def validate_trusted_model_file(path: Path) -> Path:
    """
    Validate a model path before joblib.load().

    joblib uses pickle under the hood, so model files are executable input. This
    check does not make pickle safe for untrusted artifacts; it prevents the
    most common local foot-gun where a model file is group/world writable.
    """
    resolved = path.expanduser().resolve(strict=True)
    mode = resolved.stat().st_mode
    if not stat.S_ISREG(mode):
        raise UntrustedModelError(f"model path is not a regular file: {resolved}")
    if mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise UntrustedModelError(
            f"model file is writable by group/other users: {resolved}"
        )
    return resolved
