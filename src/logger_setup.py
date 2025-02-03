import logging
import os

def get_logger(name, rundir=None, level=logging.INFO):
    """
    Creates and configures a logger that writes logs to both terminal and file.

    Args:
        name (str): The name of the logger (typically __name__).
        rundir (str): The path of the rundir.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger.
    """
    # Create logger with the specified name
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate log entries if logger is used multiple times
    if not logger.handlers:
        # Define log message format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        if rundir is not None:
            # Create file handler
            log_file = os.path.join(rundir, "run.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
