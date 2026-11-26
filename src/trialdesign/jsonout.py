"""The ``--json`` output contract.

Every command run with ``--json`` prints one JSON object to stdout with three keys besides
its results:

``schema_version``
    An integer, :data:`SCHEMA_VERSION`. Keys may be **added** within a version; removing or
    renaming a key, or changing its type, bumps the version.
``command``
    The command that produced the object (``"design"``, ``"simulate"``, ...).
``version``
    The ``trialdesign`` package version.

Non-finite numbers (NaN, infinity) are written as ``null``, so the output is strict JSON
(RFC 8259) that any parser accepts. A failing command in ``--json`` mode prints
``{"error": <message>, "code": <exit status>}`` to stderr and exits with that status.

The schema of each command is in ``docs/schema/<command>.schema.json`` (JSON Schema draft
2020-12).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["SCHEMA_VERSION", "clean", "dumps", "error_dumps"]

SCHEMA_VERSION = 1


def clean(value: Any) -> Any:
    """``value`` with numpy scalars and arrays made plain and non-finite floats as None."""
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return [clean(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        x = float(value)
        return x if math.isfinite(x) else None
    return value


def dumps(command: str, payload: Mapping[str, Any], version: str) -> str:
    """The ``--json`` text for ``command``: the versioned envelope around ``payload``."""
    body = {"schema_version": SCHEMA_VERSION, "command": command, **payload, "version": version}
    return json.dumps(clean(body), indent=2, allow_nan=False)


def error_dumps(message: str, code: int) -> str:
    """The ``--json`` error object written to stderr."""
    return json.dumps({"error": message, "code": code})
