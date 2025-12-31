"""Causal World Model (CWM) module.

Provides persistence and query capabilities for CWM states:
- CWM-A: Abstract reasoning (entities, relations, causal rules)
- CWM-G: Grounded (JEPA outputs, sensor predictions)
- CWM-E: Emotional (persona state, sentiment)
"""

from sophia.cwm.persistence import CWMPersistence

__all__ = ["CWMPersistence"]
