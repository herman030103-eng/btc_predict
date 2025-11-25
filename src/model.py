"""Определение модели — LSTM baseline. Компоновка и утилиты сохранения/загрузки.
"""
from typing import Tuple
import tensorflow as tf
from keras.layers import Attention
from tensorflow.keras import layers, models, optimizers




def build_lstm1(window: int, n_features: int, lr: float = 0.001) -> models.Model:
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


from tensorflow.keras import layers, models, optimizers
import tensorflow as tf


def build_lstm2(window: int, n_features: int, lr: float = 0.001) -> models.Model:
    inputs = layers.Input(shape=(window, n_features))

    # BiLSTM блок
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    # Attention
    attention = layers.Dense(1, activation='tanh')(x)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(128)(attention)
    attention = layers.Permute([2, 1])(attention)

    x = layers.Multiply()([x, attention])
    x = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)

    # Head
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='linear')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )
    return model


from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, LayerNormalization, GlobalAveragePooling1D, SpatialDropout1D, Conv1D
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

def build_lstm3(window: int, n_features: int, lr: float = 1e-3):
    inputs = Input(shape=(window, n_features))

    # input dropout (GPU-friendly)
    x = SpatialDropout1D(0.12)(inputs)

    # Conv stem — извлекаем локальные шаблоны
    x = Conv1D(64, kernel_size=3, padding='causal', activation='relu',
               kernel_regularizer=regularizers.l2(1e-5))(x)
    x = LayerNormalization()(x)

    # LSTM stack (без recurrent_dropout => CuDNN fast path)
    x = LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(1e-5))(x)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(1e-5))(x)
    x = Dropout(0.12)(x)

    # Multi-head attention (self-attention)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
    # residual + norm
    x = layers.Add()([x, attn])
    x = LayerNormalization()(x)

    # pooling + head
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-5))(x)
    x = Dropout(0.1)(x)
    outputs = Dense(1, activation='linear')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=lr), loss=Huber(), metrics=['mae'])
    return model

def build_lstm4(window: int, n_features: int, lr: float = 1e-3,
                        use_l2: bool = True, use_attention: bool = True,
                        loss_fn: str = 'huber'):
    reg = regularizers.l2(1e-5) if use_l2 else None
    inputs = layers.Input(shape=(window, n_features))
    x = layers.SpatialDropout1D(0.1)(inputs)
    x = layers.Conv1D(64, 3, padding='causal', activation='relu', kernel_regularizer=reg)(x)
    x = layers.LayerNormalization()(x)
    x = layers.LSTM(128, return_sequences=True, kernel_regularizer=reg)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64, return_sequences=True, kernel_regularizer=reg)(x)
    x = layers.Dropout(0.12)(x)
    if use_attention:
        att = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
        x = layers.Add()([x, att])
        x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation='linear')(x)

    model = models.Model(inputs, outputs)
    if loss_fn == 'huber':
        loss = Huber()
    elif loss_fn == 'mse':
        loss = 'mse'
    else:
        loss = loss_fn
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss=loss, metrics=['mae'])
    return model


def build_lstm5(window: int, n_features: int, lr: float = 1e-3, use_l2=True, use_attention=True):
    reg = regularizers.l2(1e-5) if use_l2 else None
    inputs = layers.Input(shape=(window, n_features))
    x = layers.SpatialDropout1D(0.1)(inputs)
    x = layers.Conv1D(64, 3, padding='causal', activation='relu', kernel_regularizer=reg)(x)
    x = layers.LayerNormalization()(x)
    x = layers.LSTM(128, return_sequences=True, kernel_regularizer=reg)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64, return_sequences=True, kernel_regularizer=reg)(x)
    x = layers.Dropout(0.12)(x)
    if use_attention:
        att = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
        x = layers.Add()([x, att])
        x = layers.LayerNormalization()(x)
    # используем последний временной шаг вместо усреднения
    x = layers.Lambda(lambda z: z[:, -1, :])(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation='linear')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='mse', metrics=['mae'])
    return model



# дополнительные архитектуры (комментарии)
# - GRU: замените LSTM на layers.GRU
# - 1D-CNN: Conv1D -> GlobalMaxPool -> Dense
# - Transformer: требуется positional encoding и больше данных