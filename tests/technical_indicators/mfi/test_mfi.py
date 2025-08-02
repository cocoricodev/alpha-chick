import numpy as np
import pandas as pd
import pytest

from tech_indicators import mfi


def test_mfi_scenarios():
    market_data = {
        "ASCENDING": {
            "high": [100.0, 102.0, 104.0, 106.0, 108.0],
            "low": [99.0, 101.0, 103.0, 105.0, 107.0],
            "close": [100.0, 102.0, 104.0, 106.0, 108.0],
            "volume": [500, 600, 700, 800, 900],
        },
        "DESCENDING": {
            "high": [108.0, 106.0, 104.0, 102.0, 100.0],
            "low": [107.0, 105.0, 103.0, 101.0, 99.0],
            "close": [108.0, 106.0, 104.0, 102.0, 100.0],
            "volume": [900, 800, 700, 600, 500],
        },
        "CONSTANT": {
            "high": [100.0, 100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0],
            "volume": [100, 100, 100, 100, 100],
        },
        "MIXED": {
            "high": [100.0, 101.0, 103.0, 105.0, 102.0],
            "low": [98.0, 99.0, 100.0, 104.0, 97.0],
            "close": [99.0, 100.0, 100.0, 102.0, 104.0],
            "volume": [500, 600, 400, 600, 400],
        },
    }

    expected_data = {
        "ASCENDING": [0.0, 0.0, 100.0, 100.0, 100.0],
        "DESCENDING": [0.0, 0.0, 0.0, 0.0, 0.0],
        "CONSTANT": [0.0, 0.0, 0.0, 0.0, 0.0],
        "MIXED": [0.0, 0.0, 100.0, 100.0, 60.623782],
    }

    for scenario_name, data in market_data.items():
        high = pd.Series(data["high"])
        low = pd.Series(data["low"])
        close = pd.Series(data["close"])
        volume = pd.Series(data["volume"])
        expected = pd.Series(expected_data[scenario_name], index=close.index)

        result = mfi(high=high, low=low, close=close, volume=volume, period=3)

        pd.testing.assert_series_equal(result, expected, atol=1e-6)


def test_mfi_raises_error_if_period_is_one():
    high = pd.Series([100, 101, 102])
    low = pd.Series([98, 99, 100])
    close = pd.Series([99, 100, 101])
    volume = pd.Series([1000, 1000, 1000])
    with pytest.raises(ValueError, match="Period must be an integer greater than 1."):
        mfi(high, low, close, volume, period=1)


def test_mfi_raises_error_if_period_is_zero():
    high = pd.Series([100, 101, 102])
    low = pd.Series([98, 99, 100])
    close = pd.Series([99, 100, 101])
    volume = pd.Series([1000, 1000, 1000])
    with pytest.raises(ValueError, match="Period must be an integer greater than 1."):
        mfi(high, low, close, volume, period=0)


def test_mfi_raises_error_if_period_is_negative():
    high = pd.Series([100, 101, 102])
    low = pd.Series([98, 99, 100])
    close = pd.Series([99, 100, 101])
    volume = pd.Series([1000, 1000, 1000])
    with pytest.raises(ValueError, match="Period must be an integer greater than 1."):
        mfi(high, low, close, volume, period=-5)


def test_mfi_raises_error_for_non_series_input():
    low = pd.Series([98, 99, 100])
    close = pd.Series([99, 100, 101])
    volume = pd.Series([1000, 1000, 1000])
    with pytest.raises(TypeError, match="Inputs 'high', 'low', 'close', and 'volume' must be pandas Series."):
        mfi(high=[100, 101, 102], low=low, close=close, volume=volume, period=3)


def test_mfi_raises_error_for_mismatched_lengths():
    high = pd.Series([100, 101, 102, 103])
    low = pd.Series([98, 99, 100, 101])
    close = pd.Series([99, 100, 101, 102])
    short_volume = pd.Series([1000, 1000])
    with pytest.raises(ValueError, match="All input Series must have the same length."):
        mfi(high, low, close, short_volume, period=3)


def test_mfi_raises_error_for_insufficient_data():
    period = 14
    short_size = 10
    high = pd.Series(range(short_size))
    low = pd.Series(range(short_size))
    close = pd.Series(range(short_size))
    volume = pd.Series(range(short_size))
    with pytest.raises(ValueError, match="Insufficient data"):
        mfi(high, low, close, volume, period=period)


def test_mfi_raises_error_for_all_nan_series():
    high = pd.Series([np.nan, np.nan, np.nan])
    low = pd.Series([np.nan, np.nan, np.nan])
    close = pd.Series([np.nan, np.nan, np.nan])
    volume = pd.Series([np.nan, np.nan, np.nan])
    with pytest.raises(ValueError, match="Input Series contain only NaN values."):
        mfi(high, low, close, volume, period=2)
