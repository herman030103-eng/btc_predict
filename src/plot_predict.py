import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Параметры
processed_data_path = "../data/processed.csv"  # сюда путь к твоему processed.csv
model_path = "../models/lstm_best.keras"
scaler_path = "../artifacts/scaler.pkl"

# Загружаем данные
df = pd.read_csv(processed_data_path)

# Получаем признаки (столбцы с окнами для close)
feat_cols = [c for c in df.columns if c.startswith("t-") and "close" in c]
X = df[feat_cols].values
y_true_scaled = df["target_close"].values.reshape(-1, 1)

# Загружаем модель и scaler
model = load_model(model_path)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Число признаков (у тебя close + индикаторы, если есть)
n_features = scaler.scale_.shape[0]

# Модель ожидает вход 3D (samples, window, features)
window = len(feat_cols) // n_features
X = X.reshape(-1, window, n_features)

# Предсказания (масштабированные)
y_pred_scaled = model.predict(X)

# Инвертируем масштабирование для первой фичи (close) для y_true и y_pred
# scaler.transform принимает массив (samples, n_features), у нас есть только close,
# поэтому заполняем остальные нулями
def inverse_transform_close(scaled_close_array):
    dummy = np.zeros((scaled_close_array.shape[0], n_features))
    dummy[:, 0] = scaled_close_array[:, 0]
    inv = scaler.inverse_transform(dummy)
    return inv[:, 0]

y_true = inverse_transform_close(y_true_scaled)
y_pred = inverse_transform_close(y_pred_scaled)

# Рисуем график
plt.figure(figsize=(14, 7))
plt.plot(y_true, label="Истинная цена")
plt.plot(y_pred, label="Предсказание")
plt.title("Сравнение истинных и предсказанных цен BTC (валидация)")
plt.xlabel("Временные шаги")
plt.ylabel("Цена BTC (USD)")
plt.legend()
plt.show()
