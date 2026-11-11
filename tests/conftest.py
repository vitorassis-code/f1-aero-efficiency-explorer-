"""Shared test fixtures: small synthetic data with known answers."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def valley_telemetry() -> pd.DataFrame:
    """A 1 km 'lap' sampled every 10 m.

    Speed is 300 km/h everywhere except a V-shaped corner centred at 500 m,
    where it drops to exactly 150 km/h. Known answers make assertions easy.
    """
    distance = np.arange(0, 1001, 10)
    speed = 300 - 150 * np.clip(1 - np.abs(distance - 500) / 100, 0, 1)
    return pd.DataFrame({"Distance": distance, "Speed": speed})


@pytest.fixture
def two_corners() -> pd.DataFrame:
    """Corner 1 at the valley (500 m) and corner 2a on the flat section (800 m)."""
    return pd.DataFrame({"Number": [1, 2], "Letter": ["", "a"], "Distance": [500.0, 800.0]})
