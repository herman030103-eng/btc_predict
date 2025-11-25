"""
src/train.py

Тренировочный скрипт: читает config.yaml, загружает processed CSV, обучает модель и сохраняет результаты.
Гарантированно завершённая версия — надёжная обработка формы данных, логирование и оценка.
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from utils import get_logger, set_seed, ensure_dir
from model import build_lstm

logger = get_logger("train")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def infer_n_features_from_processed_columns(feat_cols: list, window: int) -> int:
    """
    Попытаться надёжно вывести n_features по списку feature-колонок вида t-{k}_{feature}.
    Если длина feat_cols кратна window — используем len(feat_cols)//window.
    Иначе пытаемся распарсить суффиксы после '_' и посчитать уникальные.
    """
    if not feat_cols:
        return 1
    if len(feat_cols) % window == 0:
        return max(1, len(feat_cols) // window)

    # Парсим суффиксы (после последнего '_'), если такие есть
    suffixes = []
    for c in feat_cols:
        if "_" in c:
            suffixes.append(c.rsplit("_", 1)[-1])
        else:
            suffixes.append("close")
    uniq = list(dict.fromkeys(suffixes))
    n_feat = len(uniq)
    if n_feat > 0 and (len(feat_cols) % n_feat == 0):
        # проверяем, что это согласуется с window
        guessed_window = len(feat_cols) // n_feat
        if guessed_window == window:
            return n_feat

    # fallback: floor division
    return max(1, len(feat_cols) // window)


def load_xy_from_processed(path: str, window: int):
    """
    Загружает processed CSV и восстанавливает X (n_samples, window, n_features) и y (n_samples,).
    Ожидается, что processed.csv содержит колонки, начинающиеся с 't-' и колонку 'target_close'.
    """
    df = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c.startswith("t-")]
    if not feat_cols:
        raise ValueError(
            "Processed CSV не содержит колонок, начинающихся с 't-'. Выполните preprocess.py."
        )

    n_features = infer_n_features_from_processed_columns(feat_cols, window)
    X_flat = df[feat_cols].values

    # проверяем, что reshape возможен
    if X_flat.shape[1] != window * n_features:
        # Попробуем ещё раз определить n_features, например если названия колонок содержат лишние части
        # Но если не получается — бросаем понятную ошибку
        raise ValueError(
            f"Не удалось привести flat-матрицу к форме (n_samples, {window}, n_features). "
            f"flat width = {X_flat.shape[1]}, window = {window}, предполагаемые n_features = {n_features}."
        )

    X = X_flat.reshape((X_flat.shape[0], window, n_features))

    if "target_close" not in df.columns:
        raise ValueError("Processed CSV не содержит столбца 'target_close' (целевой признак).")

    y = df["target_close"].values

    return X, y, n_features, df


def evaluate_in_original_scale(model, X_val, y_val, n_features):
    """
    Если доступен scaler (artifacts/scaler.pkl), возвращает RMSE и MAE в оригинальном масштабе.
    Иначе возвращает None, None.
    """
    scaler_path = os.path.join("artifacts", "scaler.pkl")
    if not os.path.exists(scaler_path):
        logger.info("Scaler не найден (artifacts/scaler.pkl). Оценка в исходном масштабе пропущена.")
        return None, None

    import pickle

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Предсказания в масштабированном виде
    preds_scaled = model.predict(X_val)
    # Подготовим dummy-матрицу для inverse_transform: (n_samples, n_features)
    dummy_preds = np.zeros((preds_scaled.shape[0], n_features))
    dummy_preds[:, 0] = preds_scaled[:, 0]
    try:
        preds = scaler.inverse_transform(dummy_preds)[:, 0]
    except Exception as e:
        logger.warning(f"Не удалось выполнить inverse_transform для предсказаний: {e}")
        return None, None

    # Реальные значения
    dummy_real = np.zeros((y_val.shape[0], n_features))
    dummy_real[:, 0] = y_val
    try:
        real = scaler.inverse_transform(dummy_real)[:, 0]
    except Exception as e:
        logger.warning(f"Не удалось выполнить inverse_transform для реальных значений: {e}")
        return None, None

    rmse = np.sqrt(mean_squared_error(real, preds))
    mae = mean_absolute_error(real, preds)
    return rmse, mae


def main(config_path: str, data_csv: str = None):
    config = load_config(config_path)
    set_seed(config.get("seed", 42))
    window = int(config.get("window_size", 60))

    data_csv = data_csv or "data/processed.csv"
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"Processed data not found: {data_csv}")

    # Загружаем X, y
    X, y, n_features, df = load_xy_from_processed(data_csv, window)
    logger.info(
        f"Loaded processed data: samples={X.shape[0]}, window={window}, n_features={n_features}"
    )

    # Разделение для временных рядов (без shuffle)
    split_idx = int(X.shape[0] * float(config.get("train_split", 0.8)))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(
        f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}"
    )

    # Строим модель и тренируем
    model = build_lstm(window, n_features, lr=float(config.get("learning_rate", 0.001)))

    model_dir = config.get("model_dir", "models")
    ensure_dir(model_dir)

    checkpoint_path = os.path.join(model_dir, "lstm_best.keras")
    checkpoint = ModelCheckpoint(
        checkpoint_path, save_best_only=True, monitor="val_loss", verbose=1
    )
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)
    rl = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=int(config.get("epochs", 30)),
        batch_size=int(config.get("batch_size", 64)),
        callbacks=[checkpoint, es, rl],
        verbose=2,
    )

    final_path = os.path.join(model_dir, "lstm_final.keras")
    model.save(final_path)
    logger.info(f"Saved models: best -> {checkpoint_path}, final -> {final_path}")

    # Сохраняем историю обучения
    hist_df = pd.DataFrame(history.history)
    hist_csv = os.path.join(model_dir, "training_history.csv")
    hist_df.to_csv(hist_csv, index=False)
    logger.info(f"Training history saved to {hist_csv}")

    # Оценка в исходном масштабе (если есть scaler)
    rmse, mae = evaluate_in_original_scale(model, X_val, y_val, n_features)
    if rmse is not None:
        logger.info(f"Validation RMSE: {rmse:.6f}, MAE: {mae:.6f}")
    else:
        logger.info("Validation metrics in original scale were not computed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data", default=None, help="Path to processed CSV (optional)")
    args = parser.parse_args()
    main(args.config, args.data)
