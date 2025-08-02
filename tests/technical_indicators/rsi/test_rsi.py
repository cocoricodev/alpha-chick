import numpy as np
import pandas as pd
import pytest

from tech_indicators import rsi


def test_rsi_scenarios():
    closes_data = {
        "ASCENDING": [97, 97, 98, 100, 101],
        "CONSTANT": [100, 100, 100, 100, 100],
        "DESCENDING": [101, 100, 98, 97, 96],
        "MIXED": [100, 95, 97, 98, 94],
    }

    expected_data = {
        "ASCENDING": [50.0, 50.0, 50.0, 100.0, 100.0],
        "CONSTANT": [50.0, 50.0, 50.0, 50.0, 50.0],
        "DESCENDING": [50.0, 50.0, 50.0, 0.0, 0.0],
        "MIXED": [50.0, 50.0, 50.0, 25.925926, 15.555556],
    }

    for scenario_name, closes_values in closes_data.items():
        closes = pd.Series(closes_values)
        expected = pd.Series(expected_data[scenario_name], index=closes.index)

        result = rsi(closes=closes, period=3)

        pd.testing.assert_series_equal(result, expected, atol=1e-6)


def test_rsi_raises_error_if_period_is_one():
    with pytest.raises(ValueError, match="Period must be an integer greater than 1."):
        rsi(closes=pd.Series(range(20)), period=1)


def test_rsi_raises_error_if_period_is_zero():
    with pytest.raises(ValueError, match="Period must be an integer greater than 1."):
        rsi(closes=pd.Series(range(20)), period=0)


def test_rsi_raises_error_if_period_is_negative():
    with pytest.raises(ValueError, match="Period must be an integer greater than 1."):
        rsi(closes=pd.Series(range(20)), period=-5)


def test_rsi_raises_error_if_period_is_float():
    with pytest.raises(ValueError, match="Period must be an integer greater than 1."):
        rsi(closes=pd.Series(range(20)), period=14.5)


def test_rsi_raises_error_if_period_is_string():
    with pytest.raises(ValueError, match="Period must be an integer greater than 1."):
        rsi(closes=pd.Series(range(20)), period="fourteen")


def test_rsi_raises_error_for_non_series_input():
    with pytest.raises(TypeError, match="Input 'closes' must be a pandas Series."):
        rsi(closes=[100, 101, 102, 103], period=3)


def test_rsi_raises_error_for_insufficient_data():
    with pytest.raises(ValueError, match="Insufficient data"):
        rsi(closes=pd.Series(range(10)), period=14)


def test_rsi_raises_error_for_all_nan_series():
    with pytest.raises(ValueError, match="Input 'closes' contain only NaN values."):
        rsi(closes=pd.Series([np.nan, np.nan, np.nan]), period=2)
