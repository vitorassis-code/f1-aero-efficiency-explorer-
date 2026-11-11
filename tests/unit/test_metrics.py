"""Unit tests for exploration/metrics.py.

Fast, deterministic, no network: every test builds its own small input.
Each test follows Arrange / Act / Assert and checks one behaviour.
"""

import math
import pandas as pd
import numpy as np
import pytest

from exploration.metrics import (
    corner_label, 
    corner_min_speeds,
    count_implausible_speeds,
    filter_valid_laps,
    is_drs_open,
    min_speed_near
    
)


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



# ---------- count_implausible_speeds ----------

def test_count_implausible_speeds_flags_values_outside_physical_limits():
    #0 and 380 are exactly on the limits, so they are still plausible 
    speeds = pd.Series([-1, 0, 150, 380, 381, 999])

    assert count_implausible_speeds(speeds) == 3


def test_count_implausible_speeds_returns_zero_for_empty_series():
    speeds = pd.Series([], dtype=float)

    assert count_implausible_speeds(speeds) == 0


def test_count_implausible_speeds_respects_custom_limits():
    speeds = pd.Series([50, 100, 150])

    assert count_implausible_speeds(speeds, min_kmh=60, max_kmh=140) == 2


def test_count_implausible_speeds_does_not_count_missing_values():
    # Design decision: NaN means "missing", not "impossible"
    speeds = pd.Series([math.nan, 200])

    assert count_implausible_speeds(speeds) == 0

def test_count_implausible_speeds_does_not_count_missing_values():
    # Design decision: NaN means "missing", not "impossible".
    # 999 is included so a function that always returns 0 cannot pass.
    speeds = pd.Series([math.nan, 200, 999])

    assert count_implausible_speeds(speeds) == 1



# ---------- filter_valid_laps ----------

def test_filter_valid_laps_keeps_only_accurate_and_not_deleted():
    #Arrange
    laps = pd.DataFrame(
        {
            "Driver": ["LEC", "VER", "NOR", "PIA"],
            "IsAccurate": [True, False, True, True],
            "Deleted": [False, False, True, False],
        }
    )

    #Act
    result = filter_valid_laps(laps)

    #Assert
    assert result["Driver"].tolist() == ["LEC", "PIA"]

def test_filter_valid_laps_treats_missing_accuracy_as_not_accurate():
      # HAM is valid, so a function that always returns an empty table cannot pass
    laps = pd.DataFrame(
        {
            "Driver": ["LEC", "HAM"],
            "IsAccurate": [None, True],
            "Deleted": [False, False]
        }
    )

    result = filter_valid_laps(laps)

    assert result["Driver"].tolist() == ["HAM"]

def test_filter_valid_laps_treats_missing_deleted_as_not_deleted():
      # HAM is deleted, so a function that ignores Deleted cannot pass
    laps = pd.DataFrame(
        {
            "Driver": ["LEC", "HAM"],
            "IsAccurate": [True, True],
            "Deleted": [None, True]
        }
    )

    result = filter_valid_laps(laps)

    assert result["Driver"].tolist() == ["LEC"]

def test_filter_valid_laps_does_not_modify_the_input():
    laps = pd.DataFrame(
        {
            "Driver": ["LEC", "VER"], 
            "IsAccurate": [True, False], 
            "Deleted": [False, False]
        }
    )
    original = laps.copy()

    filter_valid_laps(laps)

    assert laps.equals(original)

def test_filter_valid_laps_treats_nan_like_none():
    # pandas often stores "missing" as NaN instead of None
    laps = pd.DataFrame(
        {
            "Driver": ["LEC", "HAM"], 
            "IsAccurate": [np.nan, True], 
            "Deleted": [np.nan, False]
        }
    )

    result = filter_valid_laps(laps)

    assert result["Driver"].tolist() == ["HAM"]

def test_filter_valid_laps_returns_empty_table_for_empty_input():
    laps = pd.DataFrame(
        {
            "Driver": [], 
            "IsAccurate": [], 
            "Deleted": []
        }
    )

    result = filter_valid_laps(laps)

    assert result.empty


def test_filter_valid_laps_raises_clear_error_when_a_column_is_missing():
    laps = pd.DataFrame(
        {
            "Driver": ["LEC"], 
            "IsAccurate": [True]
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        filter_valid_laps(laps)


# ---------- min_speed_near ----------

def test_min_speed_near_finds_the_bottom_of_the_corner(valley_telemetry):
    assert min_speed_near(valley_telemetry, distance_m=500, window_m=50) == 150


def test_min_speed_near_window_bounds_are_inclusive(valley_telemetry):
    # Window 450-470 m: the lowest point is exactly on the upper bound (470 m).
    assert min_speed_near(valley_telemetry, distance_m=460, window_m=10) == pytest.approx(195)


def test_min_speed_near_returns_none_when_window_is_outside_the_lap(valley_telemetry):
    assert min_speed_near(valley_telemetry, distance_m=5000, window_m=50) is None


@pytest.mark.parametrize("bad_window", [0, -10])
def test_min_speed_near_rejects_non_positive_window(valley_telemetry, bad_window):
    with pytest.raises(ValueError, match="window_m must be positive"):
        min_speed_near(valley_telemetry, distance_m=500, window_m=bad_window)


# ---------- corner_min_speeds ----------

def test_corner_min_speeds_returns_one_value_per_corner(valley_telemetry, two_corners):
    result = corner_min_speeds(valley_telemetry, two_corners, window_m=50)

    assert result == {"corner_1": 150.0, "corner_2a": 300.0}


def test_corner_min_speeds_skips_corners_outside_the_lap(valley_telemetry):
    # Arrange: one corner inside the lap (500 m) and one far outside (5000 m)
    corners = pd.DataFrame({"Number": [1, 9], "Letter": ["", ""], "Distance": [500.0, 5000.0]})

    # Act
    result = corner_min_speeds(valley_telemetry, corners, window_m=50)

    # Assert: only corner 1 should be in the result
    assert result == {"corner_1": 150.0}


def test_corner_min_speeds_keeps_a_legitimate_zero_speed():
    # A car stopped in a corner (0 km/h) is real data, not "missing"
    telemetry = pd.DataFrame({"Distance": [0, 10, 20], "Speed": [100, 0, 100]})
    corners = pd.DataFrame({"Number": [1], "Letter": [""], "Distance": [10.0]})

    result = corner_min_speeds(telemetry, corners, window_m=10)

    assert result == {"corner_1": 0.0}