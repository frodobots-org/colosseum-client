"""One timeout API for Python versions supported by the robot client."""
from __future__ import annotations

try:
    from asyncio import timeout
except ImportError:
    from async_timeout import timeout

__all__ = ["timeout"]
