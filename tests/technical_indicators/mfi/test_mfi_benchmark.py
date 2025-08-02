import numpy as np
import pandas as pd
import pytest

from tech_indicators import mfi


@pytest.fixture(scope="module")
def mfi_test_data() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    num_points = 1_000_000
    rng = np.random.default_rng()

    close_prices = pd.Series(rng.standard_normal(num_points).cumsum() + 100, dtype=np.float32)
    high_prices = close_prices + rng.random(num_points, dtype=np.float32)
    low_prices = close_prices - rng.random(num_points, dtype=np.float32)

    volume_data = pd.Series(rng.integers(100_000, 5_000_000, size=num_points), dtype=np.int32)

    return high_prices, low_prices, close_prices, volume_data


def test_mfi_performance(benchmark, mfi_test_data):
    high, low, close, volume = mfi_test_data

    benchmark(mfi, high=high, low=low, close=close, volume=volume, period=14)
