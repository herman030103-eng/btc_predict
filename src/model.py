"""Определение модели — LSTM baseline. Компоновка и утилиты сохранения/загрузки.
"""
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers




def build_lstm(window: int, n_features: int, lr: float = 0.001) -> models.Model:
    inputs = layers.Input(shape=(window, n_features))
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='linear')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model




# дополнительные архитектуры (комментарии)
# - GRU: замените LSTM на layers.GRU
# - 1D-CNN: Conv1D -> GlobalMaxPool -> Dense
# - Transformer: требуется positional encoding и больше данных