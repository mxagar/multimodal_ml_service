"""Custom Logger and associated helper utilities.

Author: Mikel Sagardia
Date: 2024-12-04
"""
import logging
from typing import Optional, Union
import pathlib

class Logger:
    """Simple logger class that creates a logger instance with a given name and log level.

    Usage example:

        # Create a logger instance
        # Then, we can use it (e.g., pass to a class/function)
        # and log messages with different severity levels.
        logger = Logger(name="multimodal_service_logger", log_file="logs/my_log_file.log").get_logger()

        # Log messages with different severity levels
        logger.debug("This is a debug message")
        logger.info("This is an info message")
        logger.warning("This is a warning message")
        logger.error("This is an error message")
        logger.critical("This is a critical message")
    """

    def __init__(self,
                 name: str = "multimodal_service_logger",
                 level: int = logging.INFO,
                 log_file: Optional[Union[str, pathlib.Path]] = None,
                 log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
        """Initialize the logger.

        Args:
            name (str): Logger name.
            level (int): Logging level (e.g., logging.INFO).
            log_file (Optional[Union[str, pathlib.Path]]): File to log messages.
            log_format (str): Format for log messages.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        formatter = logging.Formatter(log_format)

        if not self.logger.hasHandlers():
            # Stream handler for console output
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

        # File handler for file output
        if log_file is not None:
            if isinstance(log_file, pathlib.Path):
                log_file = str(log_file)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:  # noqa: D102
        return self.logger

    def set_level(self, level: int) -> None:  # noqa: D102
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def add_file_handler(self, log_file: Union[str, pathlib.Path]) -> None:  # noqa: D102
        if isinstance(log_file, pathlib.Path):
            log_file = str(log_file)
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def remove_handlers(self) -> None:  # noqa: D102
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

    def debug(self, message: str) -> None:  # noqa: D102
        self.logger.debug(message)

    def info(self, message: str) -> None:  # noqa: D102
        self.logger.info(message)

    def warning(self, message: str) -> None:  # noqa: D102
        self.logger.warning(message)

    def error(self, message: str) -> None:  # noqa: D102
        self.logger.error(message)

    def critical(self, message: str) -> None:  # noqa: D102
        self.logger.critical(message)
