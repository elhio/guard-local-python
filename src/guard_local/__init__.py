"""
On-device media detection for the Guard Python client.

Everything here is reached through `LocalDetectorEngine`. This engine weighs three
independent sources against each other: signed C2PA provenance, embedded metadata, and a
vision model. The Guard client looks up the engine on this package root, so it must
remain exported from here. Importing this package remains lightweight because neither
the model nor the C2PA runtime is loaded until the first call that requires them.

Quick start:

```python
import guard_local

engine = guard_local.LocalDetectorEngine()
engine.analyze(data, "image/jpeg")  # doctest: +SKIP
```
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .detection import (
    DETECTION_THRESHOLDS,
    CategoryResult,
    Signal,
    SignalMatch,
    passes_threshold,
)
from .engine import LocalDetectorEngine
from .exceptions import (
    GuardLocalError,
    MediaDecodeError,
    ModelLoadError,
    UnsupportedMediaError,
)
from .tasks import TASKS, Task

try:
    __version__ = version("guard-local-detector")
except PackageNotFoundError:  # pragma: running from an uninstalled checkout
    __version__ = "0.0.0+unknown"

__all__ = [
    "DETECTION_THRESHOLDS",
    "TASKS",
    "CategoryResult",
    "GuardLocalError",
    "LocalDetectorEngine",
    "MediaDecodeError",
    "ModelLoadError",
    "Signal",
    "SignalMatch",
    "Task",
    "UnsupportedMediaError",
    "__version__",
    "passes_threshold",
]
