"""Pure functions for aerodynamic efficiency metrics.

Rule for this module: no network, no file access, no prints.
Data goes in, results come out. That is what makes every function testable.
"""

from __future__ import annotations

import pandas as pd

# FastF1 telemetry encodes DRS state as integers; these values mean "open".
DRS_OPEN_VALUES = frozenset({10, 12, 14})

# Physical limits used to flag corrupted telemetry (km/h).
MIN_PLAUSIBLE_SPEED_KMH = 0
MAX_PLAUSIBLE_SPEED_KMH = 380


def corner_label(number: float, letter: object = "") -> str:
    """Build a stable column name for a corner, e.g. 'corner_4a'."""
    suffix = letter if isinstance(letter, str) else ""
    return f"corner_{int(number)}{suffix}"


def is_drs_open(value: object) -> bool:
    """True when a FastF1 DRS value means the flap is open."""
    return value in DRS_OPEN_VALUES