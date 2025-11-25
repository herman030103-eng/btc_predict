"""
Правильная подготовка данных без data leakage:
✅ Split по времени ДО масштабирования
✅ fit scaler только на train
✅ Отдельные окна для train и val
✅ Сохраняем два файла: processed_train.csv и processed_val.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

from utils import get_logger, ensure_dir
from features import add_indicators

logger = get_logger("preprocess")


def create_windows(values: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i+window])
        y.append(values[i+window])
    return np.array(X), np.array(y)


def preprocess(in_csv: str, out_dir: str, window: int = 60, scaler_type: str = 'minmaxNO', use_indicators: bool = True, train_split: float = 0.8):
    df = pd.read_csv(in_csv, parse_dates=['open_time'])
    df = df.sort_values('open_time').reset_index(drop=True)

    # Добавляем индикаторы
    if use_indicators:
        df = add_indicators(df)

    # Фичи
    features = ['close']
    if use_indicators:
        features += ['rsi', 'ema_12', 'ema_26', 'macd']

    data = df[features].astype(float).values

    # SPLIT ДО масштабирования ✅
    split_idx = int(len(data) * train_split)
    data_train = data[:split_idx]
    data_val = data[split_idx:]

    # scaler только на train ✅
    scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
    scaler.fit(data_train)

    # трансформация отдельно ✅
    data_train_scaled = scaler.transform(data_train)
    data_val_scaled = scaler.transform(data_val)

    # создаём окна
    X_train, y_train = create_windows(data_train_scaled, window)
    X_val, y_val = create_windows(data_val_scaled, window)

    # сохраним scaler
    ensure_dir('artifacts')
    with open('artifacts/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # сохраняем CSV
    ensure_dir(out_dir)

    def save_dataset(X, y, path):
        flat = X.reshape((X.shape[0], -1))
        cols = [f"f_{i}" for i in range(flat.shape[1])]
        out = pd.DataFrame(flat, columns=cols)
        out['target'] = y[:, 0]
        out.to_csv(path, index=False)

    save_dataset(X_train, y_train, f"{out_dir}/processed_train.csv")
    save_dataset(X_val, y_val, f"{out_dir}/processed_val.csv")

    logger.info(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")
    logger.info(f"Saved to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='data')
    parser.add_argument('--window', type=int, default=60)
    parser.add_argument('--scaler', default='minmax')
    parser.add_argument('--use_indicators', action='store_true')
    args = parser.parse_args()

    preprocess(args.input, args.output, args.window, args.scaler, args.use_indicators)