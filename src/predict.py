"""
src/predict.py

Скрипт для загрузки обученной модели и предсказания следующего закрытия BTC.
"""

import os
import yaml
import numpy as np
import pandas as pd
import argparse
import logging
import pickle

from tensorflow.keras.models import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("predict")


def load_config(path: str) -> dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_input(data_path: str, window: int, n_features: int):
    df = pd.read_csv(data_path)
    feat_cols = [c for c in df.columns if c.startswith("t-")]
    if not feat_cols:
        raise ValueError("Processed CSV не содержит колонок, начинающихся с 't-'.")

    # Каждая строка в processed.csv уже содержит все window*n_features признаков
    # Берём только последнюю строку для предсказания
    last_row = df.iloc[-1:]

    X_flat = last_row[feat_cols].values
    expected_cols = window * n_features
    if X_flat.shape[1] != expected_cols:
        raise ValueError(
            f"Input data shape {X_flat.shape}, expected (1, {expected_cols})"
        )

    X = X_flat.reshape(1, window, n_features)
    return X


def inverse_transform_prediction(pred, n_features):
    scaler_path = os.path.join("artifacts", "scaler.pkl")
    if not os.path.exists(scaler_path):
        logger.warning("Scaler не найден (artifacts/scaler.pkl). Предсказание в масштабированном виде.")
        return pred[0][0]

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    dummy = np.zeros((1, n_features))
    dummy[0, 0] = pred[0, 0]

    try:
        inv = scaler.inverse_transform(dummy)
        return inv[0, 0]
    except Exception as e:
        logger.warning(f"Не удалось выполнить обратное преобразование: {e}")
        return pred[0][0]

def infer_n_features_from_processed_columns(feat_cols: list, window: int) -> int:
    if not feat_cols:
        return 1
    if len(feat_cols) % window == 0:
        return len(feat_cols) // window
    suffixes = []
    for c in feat_cols:
        if "_" in c:
            suffixes.append(c.rsplit("_", 1)[-1])
        else:
            suffixes.append("close")
    uniq = list(dict.fromkeys(suffixes))
    n_feat = len(uniq)
    if n_feat > 0 and (len(feat_cols) % n_feat == 0):
        guessed_window = len(feat_cols) // n_feat
        if guessed_window == window:
            return n_feat
    return max(1, len(feat_cols) // window)


def main(config_path: str, data_path: str = None, model_path: str = None):
    config = load_config(config_path)
    window = int(config.get("window_size", 60))
    data_path = data_path or "data/processed.csv"
    # model_path = model_path or os.path.join(config.get("model_dir", "models"), "lstm_best.keras")
    model_path = model_path or os.path.join(config.get("model_dir", "models"), "best_model.keras")

    df = pd.read_csv(data_path)
    feat_cols = [c for c in df.columns if c.startswith("t-")]
    n_features = infer_n_features_from_processed_columns(feat_cols, window)

    logger.info(f"window = {window}, n_features = {n_features}")

    X = prepare_input(data_path, window, n_features)

    model = load_model(model_path)
    pred_scaled = model.predict(X)

    pred = inverse_transform_prediction(pred_scaled, n_features)

    logger.info(f"Next close prediction: {pred:.2f} USD")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data", default=None, help="Path to processed CSV (optional)")
    parser.add_argument("--model", default=None, help="Path to trained model (optional)")
    args = parser.parse_args()
    main(args.config, args.data, args.model)


