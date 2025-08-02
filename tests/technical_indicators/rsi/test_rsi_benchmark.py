import numpy as np
import pandas as pd
import pytest

from tech_indicators import rsi


@pytest.fixture(scope="module")
def rsi_test_data() -> pd.Series:
    num_points = 1_000_000
    rng = np.random.default_rng()
    return pd.Series(rng.standard_normal(num_points).cumsum() + 100, dtype=np.float32)


def test_rsi_performance(benchmark, rsi_test_data):
    benchmark(rsi, closes=rsi_test_data, period=14)
