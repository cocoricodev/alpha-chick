import numpy as np
import pandas as pd
import pytest

from tech_indicators import atr


def test_atr_scenarios():
    market_data = {
        "ASCENDING": {
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [100.0, 101.0, 102.0, 103.0, 104.0],
            "close": [101.0, 102.0, 103.0, 104.0, 105.0],
        },
        "DESCENDING": {
            "high": [105.0, 104.0, 103.0, 102.0, 101.0],
            "low": [104.0, 103.0, 102.0, 101.0, 100.0],
            "close": [104.0, 103.0, 102.0, 101.0, 100.0],
        },
        "CONSTANT": {
            "high": [100.0, 100.0, 100.0, 100.0, 100.0],
            "low": [100.0, 100.0, 100.0, 100.0, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0],
        },
        "MIXED": {
            "high": [100.0, 101.0, 103.0, 105.0, 102.0],
            "low": [98.0, 99.0, 100.0, 104.0, 97.0],
            "close": [99.0, 100.0, 100.0, 102.0, 104.0],
        },
    }

    expected_data = {
        "ASCENDING": [1.0, 1.0, 1.0, 1.0, 1.0],
        "DESCENDING": [1.0, 1.0, 1.0, 1.0, 1.0],
        "CONSTANT": [0.0, 0.0, 0.0, 0.0, 0.0],
        "MIXED": [2.0, 2.0, 2.333333, 3.222222, 3.814815],
    }

    for scenario_name, data in market_data.items():
        high = pd.Series(data["high"])
        low = pd.Series(data["low"])
        close = pd.Series(data["close"])
        expected = pd.Series(expected_data[scenario_name], index=high.index)

        result = atr(high=high, low=low, close=close, period=3)

        pd.testing.assert_series_equal(result, expected, atol=1e-6)


def test_atr_raises_error_if_period_is_one():
    with pytest.raises(ValueError, match="Period must be an integer greater than 1."):
        atr(high=pd.Series(range(10)), low=pd.Series(range(10)), close=pd.Series(range(10)), period=1)


def test_atr_raises_error_if_period_is_zero():
    with pytest.raises(ValueError, match="Period must be an integer greater than 1."):
        atr(high=pd.Series(range(10)), low=pd.Series(range(10)), close=pd.Series(range(10)), period=0)


def test_atr_raises_error_if_period_is_negative():
    with pytest.raises(ValueError, match="Period must be an integer greater than 1."):
        atr(high=pd.Series(range(10)), low=pd.Series(range(10)), close=pd.Series(range(10)), period=-3)


def test_atr_raises_error_if_period_is_float():
    with pytest.raises(ValueError, match="Period must be an integer greater than 1."):
        atr(high=pd.Series(range(10)), low=pd.Series(range(10)), close=pd.Series(range(10)), period=14.5)


def test_atr_raises_error_if_period_is_string():
    with pytest.raises(ValueError, match="Period must be an integer greater than 1."):
        atr(high=pd.Series(range(10)), low=pd.Series(range(10)), close=pd.Series(range(10)), period="fourteen")


def test_atr_raises_error_for_non_series_input():
    with pytest.raises(TypeError, match="Inputs 'high', 'low', and 'close' must be pandas Series."):
        atr(high=[100, 101, 102], low=pd.Series([99, 100, 101]), close=pd.Series([100, 101, 102]), period=3)


def test_atr_raises_error_for_mismatched_lengths():
    with pytest.raises(ValueError, match="All input Series must have the same length."):
        atr(high=pd.Series(range(5)), low=pd.Series(range(5)), close=pd.Series(range(3)), period=3)


def test_atr_raises_error_for_insufficient_data():
    with pytest.raises(ValueError, match="Insufficient data"):
        atr(high=pd.Series(range(5)), low=pd.Series(range(5)), close=pd.Series(range(5)), period=10)


def test_atr_raises_error_for_all_nan_series():
    nan_series = pd.Series([np.nan, np.nan, np.nan, np.nan])
    with pytest.raises(ValueError, match="Input Series contain only NaN values."):
        atr(high=nan_series, low=nan_series, close=nan_series, period=3)
