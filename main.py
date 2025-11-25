#!/usr/bin/env python3
"""
Небольшой скрипт для проверки доступности GPU в TensorFlow и простой пробный запуск на GPU.
Запуск: python check_tf_gpu.py
"""
import subprocess
import sys

try:
    import tensorflow as tf
except Exception as e:
    print("Не удалось импортировать TensorFlow:", e)
    print("Убедитесь, что TensorFlow установлен (например: pip install tensorflow).")
    sys.exit(1)

print("TensorFlow version:", tf.__version__)

# Информация о сборке (включает cuda/cudnn при наличии)
try:
    build_info = tf.sysconfig.get_build_info()
    print("Build info keys:", ", ".join(build_info.keys()))
    print(" CUDA version (build):", build_info.get("cuda_version", "unknown"))
    print(" cuDNN version (build):", build_info.get("cudnn_version", "unknown"))
except Exception:
    # Иногда get_build_info отсутствует или не содержит нужных полей
    pass

# Современный способ: list_physical_devices
gpus = tf.config.list_physical_devices("GPU")
print("Найдено физических GPU устройств:", len(gpus))
for i, gpu in enumerate(gpus):
    print(f" GPU {i} ->", gpu)

# Поддержка старой утилиты (deprecated в новых версиях)
if hasattr(tf.test, "is_gpu_available"):
    try:
        print("tf.test.is_gpu_available():", tf.test.is_gpu_available())
    except Exception:
        pass

print("tf.test.is_built_with_cuda():", tf.test.is_built_with_cuda())

# Попробуем включить memory growth (чтобы TF не захватывал всю видеопамять)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Включено memory growth для всех GPU (если поддерживается).")
    except Exception as e:
        print("Не удалось включить memory growth:", e)

# Выполним простую операцию на GPU (если он есть)
if gpus:
    try:
        with tf.device("/GPU:0"):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            c = tf.matmul(a, b)
        # Принтер результата (вызовет вычисление)
        print("Результат матричного умножения на /GPU:0:\n", c.numpy())
    except Exception as e:
        print("Ошибка при выполнении операции на GPU:", e)
else:
    print("GPU не обнаружен TensorFlow. Операция на GPU не выполняется.")

# Проверка nvidia-smi (если доступна в системе)
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        stderr=subprocess.STDOUT,
        encoding="utf-8",
    )
    print("\nnvidia-smi output:\n", out.strip())
except FileNotFoundError:
    print("\nnvidia-smi не найдена в PATH (возможно отсутствует драйвер NVIDIA или nvidia-smi не установлена).")
except subprocess.CalledProcessError as e:
    print("\nОшибка при вызове nvidia-smi:", e.output)