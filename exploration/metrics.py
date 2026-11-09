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



# Columns filter_valid_laps cannot work without.
REQUIRED_LAP_COLUMNS = ("IsAccurate", "Deleted")

def filter_valid_laps(laps: pd.DataFrame) -> pd.DataFrame: 
    """Keep laps that are accurate and not deleted.

    Missing values are treated conservatively: a missing IsAccurate
    counts as not accurate; a missing Deleted counts as not deleted.
    A missing column is a pipeline problem, not a data gap, so it raises.
    """
    missing = [col for col in REQUIRED_LAP_COLUMNS if col not in laps.columns]
    if missing:
        raise ValueError(f"Laps is missing required columns: {missing}")

    is_accurate = laps["IsAccurate"].astype("boolean").fillna(False).astype(bool)
    is_deletes = laps["Deleted"].astype("boolean").fillna(False).astype(bool)
    return laps[is_accurate & ~is_deletes]



def corner_label(number: float, letter: object = "") -> str:
    """Build a stable column name for a corner, e.g. 'corner_4a'."""
    suffix = letter if isinstance(letter, str) else ""
    return f"corner_{int(number)}{suffix}"



def is_drs_open(value: object) -> bool:
    """True when a FastF1 DRS value means the flap is open."""
    return value in DRS_OPEN_VALUES



def count_implausible_speeds(
        speeds: pd.Series, 
        min_kmh: float = MIN_PLAUSIBLE_SPEED_KMH,
        max_kmh: float = MAX_PLAUSIBLE_SPEED_KMH
) -> int: 
    """Number of speed samples outside physically plausible limits."""
    return int(((speeds < min_kmh) | (speeds > max_kmh)).sum())



