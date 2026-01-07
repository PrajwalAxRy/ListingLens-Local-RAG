"""Configuration package for RHP Analyzer.

This package provides configuration management with Pydantic BaseSettings
for type validation and automatic environment variable override support.

Environment Variables:
    Use RHP_ prefix for overrides. For nested configs use double underscore.
    Examples:
        RHP_PATHS__OUTPUT_DIR=/custom/output
        RHP_LLM__TEMPERATURE=0.5
        RHP_AGENTS__PARALLEL_EXECUTION=false
        HF_TOKEN=your_hugging_face_token
"""

from .schema import (
    AgentsConfig,
    AppConfig,
    IngestionConfig,
    LLMConfig,
    LoggingConfig,
    PathsConfig,
    ReportingConfig,
)

__all__ = [
    "AppConfig",
    "PathsConfig",
    "LLMConfig",
    "IngestionConfig",
    "AgentsConfig",
    "ReportingConfig",
    "LoggingConfig",
]
