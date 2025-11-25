import os
import argparse
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from utils import get_logger, set_seed, ensure_dir
from model import build_lstm1

import pickle

logger = get_logger("train")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(path: str, window: int):
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c.startswith("f_")]

    if not feat_cols:
        raise ValueError(f"{path} не содержит фич-колонок вида f_*")

    flat = df[feat_cols].values

    if flat.shape[1] % window != 0:
        raise ValueError(f"Нельзя reshape: width={flat.shape[1]}, window={window}")

    n_features = flat.shape[1] // window
    X = flat.reshape((-1, window, n_features))

    if "target" not in df.columns:
        raise ValueError(f"{path} не содержит колонки target")

    y = df["target"].values

    return X, y, n_features


def inverse_scale(preds, y, n_features):
    scaler_path = os.path.join("artifacts", "scaler.pkl")
    if not os.path.exists(scaler_path):
        return None, None

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    dummy_preds = np.zeros((preds.shape[0], n_features))
    dummy_preds[:, 0] = preds[:, 0]

    dummy_real = np.zeros((y.shape[0], n_features))
    dummy_real[:, 0] = y

    try:
        real_preds = scaler.inverse_transform(dummy_preds)[:, 0]
        real_y = scaler.inverse_transform(dummy_real)[:, 0]
    except Exception as e:
        logger.warning(f"Ошибка inverse_transform: {e}")
        return None, None

    rmse = np.sqrt(mean_squared_error(real_y, real_preds))
    mae = mean_absolute_error(real_y, real_preds)

    return rmse, mae


def main(config_path: str):
    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    window = int(config.get("window_size", 60))
    model_dir = config.get("model_dir", "models")
    ensure_dir(model_dir)

    train_csv = "data/processed/processed_train.csv"
    val_csv = "data/processed/processed_val.csv"

    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError("processed_train.csv или processed_val.csv не найдены! Запусти preprocess.py.")

    X_train, y_train, n_features_train = load_dataset(train_csv, window)
    X_val, y_val, n_features_val = load_dataset(val_csv, window)

    if n_features_train != n_features_val:
        raise ValueError("n_features в train и val не совпадают")

    n_features = n_features_train

    logger.info(f"Train samples: {X_train.shape[0]}")
    logger.info(f"Val samples:   {X_val.shape[0]}")
    logger.info(f"window={window}, n_features={n_features}")

    checkpoint_path = os.path.join(model_dir, "lstm_best.keras")

    # Попытка загрузить существующую модель
    if os.path.exists(checkpoint_path):
        logger.info(f"Загружаем существующую модель из {checkpoint_path} для дообучения")
        model = load_model(checkpoint_path)
    else:
        logger.info("Создаём новую модель")
        model = build_lstm1(window, n_features, lr=float(config.get("learning_rate", 0.001)))

    checkpoint = ModelCheckpoint(
        checkpoint_path, save_best_only=True, monitor="val_loss", verbose=1
    )

    es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=1)
    rl = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, verbose=1)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=int(config.get("epochs", 30)),
        batch_size=int(config.get("batch_size", 64)),
        callbacks=[checkpoint, es, rl],
        verbose=2,
        shuffle=False,
    )

    final_path = os.path.join(model_dir, "lstm_final.keras")
    model.save(final_path)
    logger.info(f"Saved models: best -> {checkpoint_path}, final -> {final_path}")

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(model_dir, "training_history.csv"), index=False)

    preds_scaled = model.predict(X_val)
    rmse, mae = inverse_scale(preds_scaled, y_val, n_features)

    if rmse is not None:
        logger.info(f"Validation RMSE: {rmse:.2f} USD, MAE: {mae:.2f} USD")
    else:
        logger.info("Не удалось посчитать метрики в оригинальном масштабе")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
