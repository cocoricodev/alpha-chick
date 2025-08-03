import numpy as np
import pandas as pd
import pytest

from tech_indicators import atr


@pytest.fixture(scope="module")
def atr_test_data() -> tuple[pd.Series, pd.Series, pd.Series]:
    num_points = 1_000_000
    rng = np.random.default_rng()

    close_prices = pd.Series(rng.standard_normal(num_points).cumsum() + 100, dtype=np.float32)
    high_prices = close_prices + rng.random(num_points, dtype=np.float32)
    low_prices = close_prices - rng.random(num_points, dtype=np.float32)

    return high_prices, low_prices, close_prices


def test_atr_performance(benchmark, atr_test_data):
    high, low, close = atr_test_data
    benchmark(atr, high=high, low=low, close=close, period=14)
