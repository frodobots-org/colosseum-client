"""One timeout API for the supported Python versions.

The robot-side environment is Python 3.10 because it contains the validated
DROID/ZED bindings.  Python 3.11 exposes ``asyncio.timeout``; the older
environment uses the compatible ``async_timeout`` package already installed
with that environment.
"""
from __future__ import annotations

try:  # Python 3.11+
    from asyncio import timeout
except ImportError:  # Python 3.10
    from async_timeout import timeout

__all__ = ["timeout"]
