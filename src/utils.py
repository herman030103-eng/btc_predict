"""Утилиты: логирование, сохранение/загрузка, seed."""
import os
import random
import logging
from typing import Any
import numpy as np
import tensorflow as tf




def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)




def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)




def get_logger(name: str = __name__) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger