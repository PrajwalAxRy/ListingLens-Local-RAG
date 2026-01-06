import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """
    Configure logging for the application using Loguru.

    Args:
        log_level: The logging level for the console (default: "INFO")
        log_dir: Directory to store log files (default: "logs")
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Define log format
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    # Console Sink (Colorized)
    logger.add(sys.stderr, format=log_format, level=log_level, colorize=True)

    # File Sink (Daily Rotation, 30 days retention)
    # Using format from blueprint/milestones: logs/YYYY-MM-DD.log
    logger.add(
        log_path / "{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",  # Capture everything in file
        encoding="utf-8",
    )

    # Error Sink (Separate file for errors)
    logger.add(
        log_path / "errors.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",  # Rotate errors if they get too big
        retention="30 days",
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
    )

    logger.info(f"Logging initialized. Level: {log_level}, Dir: {log_path}")


# Initialize with defaults if run directly
if __name__ == "__main__":
    setup_logging()
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
