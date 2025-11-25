"""Расчёт технических индикаторов (опционально).
Использую библиотеку `ta` (https://pypi.org/project/ta/)
"""
import pandas as pd


try:
    import ta
except Exception:
    raise ImportError("Install `ta` to use technical indicators: pip install ta")




def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if ta is None:
        # raise ImportError("Install `ta` to use technical indicators: pip install ta")
        print("error")
    out = df.copy()
    close = out['close']
    out['rsi'] = ta.momentum.rsi(close, window=14)
    out['ema_12'] = ta.trend.ema_indicator(close, window=12)
    out['ema_26'] = ta.trend.ema_indicator(close, window=26)
    out['macd'] = out['ema_12'] - out['ema_26']
    out = out.fillna(method='bfill').fillna(method='ffill')
    return out