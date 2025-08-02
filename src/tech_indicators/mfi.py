import pandas as pd


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int) -> pd.Series:
    """
    :param high: A pandas Series of high prices.
    :param low: A pandas Series of low prices.
    :param close: A pandas Series of closing prices.
    :param volume: A pandas Series of trading volumes.
    :param period: The number of periods to use for the MFI calculation.
    :return: A pandas Series containing the MFI values.
    """
    if not isinstance(period, int) or period <= 1:
        raise ValueError("Period must be an integer greater than 1.")

    if (
        not isinstance(high, pd.Series)
        or not isinstance(low, pd.Series)
        or not isinstance(close, pd.Series)
        or not isinstance(volume, pd.Series)
    ):
        raise TypeError("Inputs 'high', 'low', 'close', and 'volume' must be pandas Series.")

    if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
        raise ValueError("All input Series must have the same length.")

    if len(high) < period + 1:
        raise ValueError(f"Insufficient data: length ({len(high)}) must be >= period + 1 ({period + 1}).")

    if high.isna().all() or low.isna().all() or close.isna().all() or volume.isna().all():
        raise ValueError("Input Series contain only NaN values.")

    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume

    change = typical_price.diff()

    positive_flow = raw_money_flow.where(change > 0, 0.0)
    negative_flow = raw_money_flow.where(change < 0, 0.0)

    roll_pos = positive_flow.rolling(window=period, min_periods=period).sum()
    roll_neg = negative_flow.rolling(window=period, min_periods=period).sum()

    shift_pos = positive_flow.shift(period - 1)
    shift_neg = negative_flow.shift(period - 1)

    pos_sum = roll_pos - shift_pos.fillna(0.0)
    neg_sum = roll_neg - shift_neg.fillna(0.0)

    ratio = (pos_sum / neg_sum).fillna(0.0)

    mfi_values = 100.0 - (100.0 / (1.0 + ratio))

    return pd.Series(mfi_values, index=high.index)
