"""Unit tests for exploration/metrics.py.

Fast, deterministic, no network: every test builds its own small input.
Each test follows Arrange / Act / Assert and checks one behaviour.
"""

import math

import numpy as np
import pytest

from exploration.metrics import corner_label, is_drs_open


# ---------- corner_label ----------

@pytest.mark.parametrize(
    ("number", "letter", "expected"),
    [
        (3, "", "corner_3"),
        (4, "a", "corner_4a"),
        (5, math.nan, "corner_5"),   # FastF1 may return NaN for "no letter"
        (6.0, None, "corner_6"),     # numbers can arrive as floats
    ],
)
def test_corner_label(number, letter, expected):
    assert corner_label(number, letter) == expected


# ---------- is_drs_open ----------

@pytest.mark.parametrize(
    ("value", "expected"),
    [(10, True), (12, True), (14, True), (np.int64(12), True), (0, False), (8, False), (math.nan, False)],
)
def test_is_drs_open(value, expected):
    assert is_drs_open(value) is expected