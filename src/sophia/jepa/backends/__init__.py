"""JEPA backend implementations.

This module provides pluggable backends for JEPA operations:
- StubJEPABackend: CPU-friendly stub for tests/CI (default)
- PoCJEPABackend: PoC backend with real V-JEPA model (requires GPU + weights)
"""

from sophia.jepa.backends.poc import PoCJEPABackend

__all__ = ["PoCJEPABackend"]
