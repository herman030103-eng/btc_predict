"""Подготовка данных: нормализация, окно, train/test split.
Скрипт создаёт CSV с колонками: window -> target
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import get_logger, ensure_dir
from features import add_indicators


logger = get_logger("preprocess")




def create_windows(values: np.ndarray, window: int):
    X = []
    y = []
    for i in range(len(values) - window):
        X.append(values[i:i+window])
        y.append(values[i+window])
    return np.array(X), np.array(y)




def preprocess(in_csv: str, out_csv: str, window: int = 60, scaler_type: str = 'minmax', use_indicators: bool = True):
    df = pd.read_csv(in_csv, parse_dates=['open_time'])
    df = df.sort_values('open_time').reset_index(drop=True)


# опционально добавляем индикаторы
    if use_indicators:
        df = add_indicators(df)


# Выбираем фичи. Для начала — close. Можно расширить.
    features = ['close']
    if use_indicators:
        features += ['rsi', 'ema_12', 'ema_26', 'macd']


    data = df[features].astype(float).values


# масштабирование
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)


# создаём окна на основе только цены или на основе всех фич
    X, y = create_windows(data_scaled, window)


# сохраняем scaler вместе с данными (pickle)
    import pickle
    ensure_dir('artifacts')
    with open('artifacts/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


# Сохраняем как CSV: X as flattened + y
    n_features = data.shape[1]
    columns = []
    for i in range(window):
        for j in range(n_features):
            columns.append(f"t-{window-i}_{features[j]}")
    columns.append('target_close')


    flat = X.reshape((X.shape[0], X.shape[1]*X.shape[2]))
    out = pd.DataFrame(flat, columns=columns[:-1])
    out['target_close'] = y[:, 0] # target is scaled close
    out['index_time'] = df['open_time'].iloc[window:].values


    out.to_csv(out_csv, index=False)
    logger.info(f"Saved processed data to {out_csv}. Rows: {len(out)}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--window', type=int, default=60)
    parser.add_argument('--scaler', default='minmax')
    parser.add_argument('--use_indicators', action='store_true')
    args = parser.parse_args()
    preprocess(args.input, args.output, args.window, args.scaler, args.use_indicators)