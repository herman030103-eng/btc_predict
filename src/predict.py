"""
predict.py — предсказание следующего интервала по RAW данным
"""
import argparse
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from features import add_indicators


def load_last_window(raw_csv: str, window: int):
    df = pd.read_csv(raw_csv, parse_dates=["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)

    # Добавляем индикаторы как в preprocess.py
    df = add_indicators(df)

    features = ['close', 'rsi', 'ema_12', 'ema_26', 'macd']
    data = df[features].astype(float).values

    if len(data) < window:
        raise ValueError(f"Недостаточно данных: нужно {window}, есть {len(data)}")

    # Берём последние window строк
    last_window = data[-window:]
    return last_window, len(features)


def scale_window(window_data: np.ndarray):
    with open('artifacts/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    scaled = scaler.transform(window_data)
    return scaled, scaler


def predict_next(raw_csv: str, model_path: str, window: int):
    # 1. Формируем окно
    last_window, n_features = load_last_window(raw_csv, window)

    # 2. Масштабируем тем же scaler'ом, что на обучении
    scaled_window, scaler = scale_window(last_window)

    # 3. Делаем форму (1, window, n_features)
    X = scaled_window.reshape(1, window, n_features)

    # 4. Грузим модель
    model = load_model(model_path)

    # 5. Предикт в масштабе scaler'а
    pred_scaled = model.predict(X)[0][0]

    # 6. Обратное масштабирование только первого признака (close)
    dummy = np.zeros((1, n_features))
    dummy[0, 0] = pred_scaled
    pred_real = scaler.inverse_transform(dummy)[0][0]

    return pred_real


def main(raw_csv: str, model_path: str, window: int):
    pred = predict_next(raw_csv, model_path, window)
    print(f"\n✅ Прогноз следующего часа: {pred:.2f} USD\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Path to raw CSV")
    parser.add_argument("--model", default="models/lstm_best.keras")
    parser.add_argument("--window", type=int, default=60)
    args = parser.parse_args()

    main(args.raw, args.model, args.window)
