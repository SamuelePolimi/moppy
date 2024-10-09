import tensorflow as tf
import os
from datetime import datetime


class TensorBoardLogger:
    _instance = None

    def __new__(cls, log_dir="logs", enabled=True):
        if cls._instance is None:
            cls._instance = super(TensorBoardLogger, cls).__new__(cls)
            # Create a unique log directory for this instance
            cls._log_dir = os.path.join(log_dir, f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
            os.makedirs(cls._log_dir, exist_ok=True)
            cls._instance.summary_writer = tf.summary.create_file_writer(cls._log_dir)
        return cls._instance

    def log_metrics(self, step, metrics, log_list_name='train'):
        if not isinstance(metrics, dict):
            raise ValueError("Metrics should be a dictionary.")
        if not all(isinstance(key, str) for key in metrics.keys()):
            raise ValueError("Keys of metrics should be strings.")
        if not all(isinstance(value, (int, float)) for value in metrics.values()):
            raise ValueError("Values of metrics should be integers or floats.")
        if self._instance is None:
            raise ValueError("TensorBoardLogger is not initialized.")
        if self._instance.summary_writer is None:
            raise ValueError("TensorBoardLogger is not initialized.")

        with self._instance.summary_writer.as_default():
            for key, value in metrics.items():
                tf.summary.scalar(f"{log_list_name}/{key}", value, step=step)
