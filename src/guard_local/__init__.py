"""
On-device media detection for the Guard Python client.

Everything here is reached through `LocalDetectorEngine`. The Guard client looks
it up on this package root, so it must stay exported from here, and importing this
package must stay cheap, since no model is loaded until the first call.

Quick start:

```python
import guard_local

engine = guard_local.LocalDetectorEngine()
engine.analyze(data, "image/jpeg") # doctest: +SKIP
```
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .engine import LocalDetectorEngine
from .exceptions import (
    GuardLocalError,
    MediaDecodeError,
    ModelLoadError,
    UnsupportedMediaError,
)
from .models import TASKS, Task

try:
    __version__ = version("guard-local-detector")
except PackageNotFoundError:  # pragma: running from an uninstalled checkout
    __version__ = "0.0.0+unknown"

__all__ = [
    "TASKS",
    "GuardLocalError",
    "LocalDetectorEngine",
    "MediaDecodeError",
    "ModelLoadError",
    "Task",
    "UnsupportedMediaError",
    "__version__",
]
