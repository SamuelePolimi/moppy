def get_tensorboard_logger():
    try:
        import tensorflow as tf
        from .tensorboard_logger import TensorBoardLogger
        return TensorBoardLogger
    except ImportError:
        return None

from .default_logger import Logger
