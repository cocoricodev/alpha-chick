# tech_indicators/atr.py
import numpy as np
import pandas as pd


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    :param high: pandas Series of high prices
    :param low: pandas Series of low  prices
    :param close: pandas Series of close prices
    :param period: look-back window (>1)
    :return: pandas Series with ATR values
    """
    if not isinstance(period, int) or period <= 1:
        raise ValueError("Period must be an integer greater than 1.")

    if not all(isinstance(s, pd.Series) for s in (high, low, close)):
        raise TypeError("Inputs 'high', 'low', and 'close' must be pandas Series.")

    if not (len(high) == len(low) == len(close)):
        raise ValueError("All input Series must have the same length.")

    if len(high) < period + 1:
        raise ValueError(f"Insufficient data: length ({len(high)}) must be >= period + 1 ({period + 1}).")

    if high.isna().all() or low.isna().all() or close.isna().all():
        raise ValueError("Input Series contain only NaN values.")

    prev_close = close.shift(1).fillna(close)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = np.maximum(np.maximum(tr1, tr2), tr3)

    atr_values = true_range.ewm(alpha=1 / period, adjust=False).mean()

    return pd.Series(atr_values, index=high.index)
