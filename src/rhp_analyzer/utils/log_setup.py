import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

# ============================================================================
# Context Management Constants
# ============================================================================


class LogPhases:
    """
    Predefined constants for processing phases to ensure consistency across
    the multi-agent workflow. These prevent typos in phase names.

    Based on the 6-phase processing pipeline from blueprint.md:
    1. Ingestion & Preprocessing
    2. Chunking & Embedding
    3. Analysis (multi-agent)
    4. Critique & Verification
    5. Report Generation
    6. Completion
    """

    INITIALIZATION = "initialization"
    INGESTION = "ingestion"
    PREPROCESSING = "preprocessing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    ANALYSIS = "analysis"
    CRITIQUE = "critique"
    REPORTING = "reporting"
    COMPLETION = "completion"


class AgentNames:
    """
    Predefined constants for agent names in the multi-agent analysis system.
    These ensure consistent naming across the 14-agent intelligence tier.

    Based on agents defined in blueprint.md:
    - Analysis Agents: Business, Industry, Management, Capital Structure,
      Forensic, Red Flag, Governance, Legal, Valuation, Utilization, Promoter DD
    - Validation Agents: Self-Critic, Summarizer
    - Decision Agent: Investment Committee
    - On-Demand Agent: Q&A
    """

    ARCHITECT = "ArchitectAgent"
    BUSINESS_ANALYST = "BusinessAnalystAgent"
    INDUSTRY_ANALYST = "IndustryAnalystAgent"
    MANAGEMENT = "ManagementAgent"
    CAPITAL_STRUCTURE = "CapitalStructureAgent"
    FORENSIC_ACCOUNTANT = "ForensicAccountantAgent"
    RED_FLAG = "RedFlagAgent"
    GOVERNANCE = "GovernanceAgent"
    LEGAL = "LegalAgent"
    VALUATION = "ValuationAgent"
    UTILIZATION = "UtilizationAgent"
    PROMOTER_DD = "PromoterDueDiligenceAgent"
    SELF_CRITIC = "SelfCriticAgent"
    SUMMARIZER = "SummarizerAgent"
    INVESTMENT_COMMITTEE = "InvestmentCommitteeAgent"
    QA_AGENT = "QAAgent"


# Valid context field names
VALID_CONTEXT_FIELDS = {"document_id", "phase", "agent_name"}


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
    # Handle None log_dir by using default
    if log_dir is None:
        log_dir = "logs"

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
def log_context(**kwargs) -> Generator[None, None, None]:
    """
    Context manager to bind contextual information to logs using Loguru's
    contextualize() method. This ensures thread-safe context isolation using
    Python's contextvars.

    This function adds structured metadata to log messages within its scope,
    allowing logs to be tagged with document IDs, processing phases, and agent
    names throughout the multi-agent workflow.

    Supported Context Fields:
        document_id (str): Unique identifier for the RHP document being processed
        phase (str): Current processing phase (see LogPhases class for constants)
        agent_name (str): Name of the active analysis agent (see AgentNames class)

    Args:
        **kwargs: Key-value pairs for context fields. Field names are validated
                  against VALID_CONTEXT_FIELDS, but values are flexible to support
                  runtime-generated identifiers.

    Raises:
        ValueError: If an invalid field name is provided (not in VALID_CONTEXT_FIELDS)

    Usage Examples:

        # Basic usage with document tracking
        with log_context(document_id="DOC-20260107-001", phase=LogPhases.INGESTION):
            logger.info("Starting PDF extraction")
            # All logs here include document_id and phase

        # Multi-agent workflow example
        with log_context(document_id="RHP-123", phase=LogPhases.ANALYSIS):
            logger.info("Beginning multi-agent analysis")

            # Nested context for specific agent
            with log_context(agent_name=AgentNames.FORENSIC_ACCOUNTANT):
                logger.info("Analyzing financial statements")
                logger.warning("CFO/EBITDA ratio below 50%")

            # Context resets after nested block
            with log_context(agent_name=AgentNames.RED_FLAG):
                logger.info("Scanning for red flags")

        # Phase transitions
        with log_context(document_id="RHP-456", phase=LogPhases.REPORTING):
            logger.info("Generating markdown report")
            with log_context(agent_name=AgentNames.INVESTMENT_COMMITTEE):
                logger.info("Calculating final scorecard")

    Context Isolation:
        Contexts are properly isolated and don't leak between operations:
        - Exiting a context removes its fields from subsequent logs
        - Nested contexts combine fields from all active scopes
        - Thread-safe: Each thread maintains independent context

    Implementation Note:
        Uses Loguru's contextualize() method which leverages Python's contextvars
        for thread-local context management, ensuring no context leakage across
        concurrent operations or workflow phases.
    """
    # Validate field names (values are intentionally flexible)
    invalid_fields = set(kwargs.keys()) - VALID_CONTEXT_FIELDS
    if invalid_fields:
        raise ValueError(f"Invalid context field(s): {invalid_fields}. " f"Valid fields are: {VALID_CONTEXT_FIELDS}")

    with logger.contextualize(**kwargs):
        yield


# Initialize with defaults if run directly
if __name__ == "__main__":
    setup_logging()
    logger.info("This is an info message")

    # Test basic context with constants
    with log_context(document_id="DOC-999", phase=LogPhases.INGESTION):
        logger.info("This log is inside a context")
        logger.warning("Warning inside context")

        # Test nested context with agent name
        with log_context(agent_name=AgentNames.FORENSIC_ACCOUNTANT):
            logger.info("Nested context with agent name")

    # Test context isolation - should not have any context fields
    logger.error("This is an error message outside context")

    # Test validation: this should raise ValueError
    try:
        with log_context(invalid_field="test"):
            logger.info("This should not appear")
    except ValueError as e:
        logger.info(f"Validation working correctly: {e}")
