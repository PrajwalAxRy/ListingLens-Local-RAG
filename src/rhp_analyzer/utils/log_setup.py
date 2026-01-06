import sys
from contextlib import contextmanager
from pathlib import Path

from loguru import logger


def format_record(record):
    """
    Custom format function to include context fields if present.
    """
    format_string = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"

    # Add context fields if they exist in extra
    if record["extra"].get("document_id"):
        format_string += " | <yellow>DOC:{extra[document_id]}</yellow>"
    if record["extra"].get("phase"):
        format_string += " | <magenta>{extra[phase]}</magenta>"
    if record["extra"].get("agent_name"):
        format_string += " | <blue>{extra[agent_name]}</blue>"

    format_string += " - <level>{message}</level>\n"

    if record["exception"]:
        format_string += "{exception}\n"

    return format_string


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

    # Console Sink (Colorized) using custom formatter
    logger.add(sys.stderr, format=format_record, level=log_level, colorize=True)

    # File Sink (Daily Rotation, 30 days retention)
    # Using format from blueprint/milestones: logs/YYYY-MM-DD.log
    logger.add(
        log_path / "{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {extra} - {message}",
        level="DEBUG",  # Capture everything in file
        encoding="utf-8",
    )

    # Error Sink (Separate file for errors)
    logger.add(
        log_path / "errors.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {extra} - {message}",
        rotation="10 MB",  # Rotate errors if they get too big
        retention="30 days",
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
    )

    logger.info(f"Logging initialized. Level: {log_level}, Dir: {log_path}")


@contextmanager
def log_context(**kwargs):
    """
    Context manager to bind contextual information to logs.

    Usage:
        with log_context(document_id="DOC123", phase="ingestion"):
            logger.info("Processing document")
    """
    with logger.contextualize(**kwargs):
        yield


# Initialize with defaults if run directly
if __name__ == "__main__":
    setup_logging()
    logger.info("This is an info message")

    with log_context(document_id="DOC-999", phase="TEST_PHASE"):
        logger.info("This logic is inside a context")
        logger.warning("Warning inside context")

        with log_context(agent_name="TestAgent"):
            logger.info("Nested context with agent name")

    logger.error("This is an error message outside context")
