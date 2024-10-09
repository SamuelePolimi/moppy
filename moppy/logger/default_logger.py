
class Logger:
    def __init__(self, log_dir="logs", logging_enabled=False):
        self._logging_enabled = logging_enabled
        self._tensorboard_logger = None
        if logging_enabled:
            self.enable_logging(log_dir)

    def log_metrics(self, step, metrics, log_list_name='train'):
        """
        Log metrics to tensorboard if logging is enabled, does nothing otherwise.
        Args:
            step: current training step
            metrics: dictionary containing the metrics to log
            log_list_name: name of the list to log the metrics to
        """

        if not self._logging_enabled:
            return
        self._tensorboard_logger.log_metrics(step, metrics, log_list_name)

    def enable_logging(self, log_dir="logs"):
        """
        Enable logging to tensorboard.
        Args:
            log_dir: directory to save the logs to
        """
        self._logging_enabled = True
        if self._tensorboard_logger is None:
            from moppy.logger import get_tensorboard_logger
            TensorBoardLogger = get_tensorboard_logger()
            if TensorBoardLogger is None:
                print("TensorFlow is not installed. Logging is disabled.")
                while True:
                    response = input("Do you want to install TensorFlow now[pip install tensorflow]? (y/n): ")
                    if response.lower() == 'y':
                        import os
                        os.system('pip install tensorflow')
                        print("TensorFlow installed.")
                        self.enable_logging(log_dir)
                        break
                    elif response.lower() == 'n':
                        while True:
                            response = input("Do you want to continue without logging? (y/n): ")
                            if response.lower() == 'y':
                                self._logging_enabled = False
                                break
                            elif response.lower() == 'n':
                                exit()
                            else:
                                print("Invalid input. Please enter 'y' or 'n'.")
                    else:
                        print("Invalid input. Please enter 'y' or 'n'.")

            else:
                self._tensorboard_logger = TensorBoardLogger(log_dir, self._logging_enabled)
                print(f"Logging enabled. Logs will be saved in {self._tensorboard_logger._log_dir}")
