"""Configuration schema using Pydantic BaseSettings with environment variable support.

This module defines the configuration schema for the RHP Analyzer application.
Environment variables can be used to override config values using the RHP_ prefix.
For example: RHP_LLM__TEMPERATURE=0.2 or RHP_PATHS__OUTPUT_DIR="/custom/output"
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PathsConfig(BaseSettings):
    """File system paths configuration."""

    model_config = SettingsConfigDict(env_prefix="RHP_PATHS__", env_nested_delimiter="__")

    input_dir: Path = Field(default="./data/input", description="Directory for input RHP PDFs")
    output_dir: Path = Field(default="./data/output", description="Directory for generated reports")
    logs_dir: Path = Field(default="./logs", description="Directory for log files")
    data_dir: Path = Field(default="./data", description="Base data directory")

    @field_validator("input_dir", "output_dir", "logs_dir", "data_dir", mode="before")
    @classmethod
    def validate_paths(cls, v):
        """Convert string paths to Path objects and resolve them."""
        if isinstance(v, str):
            return Path(v).expanduser().resolve()
        return v


class LLMConfig(BaseSettings):
    """Large Language Model configuration."""

    model_config = SettingsConfigDict(env_prefix="RHP_LLM__", env_nested_delimiter="__")

    provider: Literal["huggingface"] = Field(default="huggingface", description="LLM provider")
    context_model: str = Field(default="Qwen/Qwen2.5-32B-Instruct", description="Model for large context processing")
    reasoning_model: str = Field(
        default="meta-llama/Llama-3.3-70B-Instruct", description="Model for complex reasoning tasks"
    )
    api_key: str | None = Field(default=None, description="Hugging Face API token")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(default=4096, gt=0, description="Maximum tokens per request")
    timeout: int = Field(default=120, gt=0, description="Request timeout in seconds")

    @field_validator("api_key", mode="before")
    @classmethod
    def validate_api_key(cls, v):
        """Load API key from environment if not provided."""
        if v is None:
            import os

            return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        return v


class IngestionConfig(BaseSettings):
    """Document ingestion and processing configuration."""

    model_config = SettingsConfigDict(env_prefix="RHP_INGESTION__", env_nested_delimiter="__")

    chunk_size: int = Field(default=1000, gt=0, description="Target size for text chunks in tokens")
    chunk_overlap: int = Field(default=100, ge=0, description="Overlap between chunks in tokens")
    min_chunk_size: int = Field(default=200, gt=0, description="Minimum chunk size in tokens")
    batch_size: int = Field(default=32, gt=0, description="Batch size for embedding generation")

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v, info):
        """Ensure chunk overlap is less than chunk size."""
        if info.data and "chunk_size" in info.data:
            chunk_size = info.data["chunk_size"]
            if v >= chunk_size:
                raise ValueError(f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})")
        return v

    @field_validator("min_chunk_size")
    @classmethod
    def validate_min_chunk_size(cls, v, info):
        """Ensure minimum chunk size is reasonable."""
        if info.data and "chunk_size" in info.data:
            chunk_size = info.data["chunk_size"]
            if v > chunk_size:
                raise ValueError(f"min_chunk_size ({v}) cannot be greater than chunk_size ({chunk_size})")
        return v


class AgentsConfig(BaseSettings):
    """Agent system configuration."""

    model_config = SettingsConfigDict(env_prefix="RHP_AGENTS__", env_nested_delimiter="__")

    enabled: list[str] = Field(
        default=["architect", "forensic", "red_flag", "governance", "legal", "summarizer", "critic"],
        description="List of enabled analysis agents",
    )
    max_revisions: int = Field(default=2, ge=0, le=5, description="Maximum revision cycles")
    parallel_execution: bool = Field(default=True, description="Enable parallel agent execution")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Default agent temperature")
    max_tokens: int = Field(default=4096, gt=0, description="Default max tokens per agent")

    @field_validator("enabled")
    @classmethod
    def validate_enabled_agents(cls, v):
        """Validate that enabled agents are from the known set."""
        valid_agents = {
            "architect",
            "business",
            "industry",
            "management",
            "capital_structure",
            "forensic",
            "red_flag",
            "governance",
            "legal",
            "valuation",
            "utilization",
            "promoter_dd",
            "summarizer",
            "critic",
            "qa",
            "investment_committee",
        }
        for agent in v:
            if agent not in valid_agents:
                raise ValueError(f"Unknown agent: {agent}. Valid agents: {sorted(valid_agents)}")
        return v


class ReportingConfig(BaseSettings):
    """Report generation configuration."""

    model_config = SettingsConfigDict(env_prefix="RHP_REPORTING__", env_nested_delimiter="__")

    formats: list[Literal["markdown", "pdf", "json"]] = Field(
        default=["markdown", "pdf"], description="Output formats to generate"
    )
    template: str = Field(default="default", description="Report template name")
    include_appendices: bool = Field(default=True, description="Include detailed appendices")
    template_path: Path | None = Field(default=None, description="Custom template directory")

    @field_validator("template_path", mode="before")
    @classmethod
    def validate_template_path(cls, v):
        """Convert template path to Path object if provided."""
        if v is not None and isinstance(v, str):
            return Path(v).expanduser().resolve()
        return v


class LoggingConfig(BaseSettings):
    """Logging system configuration."""

    model_config = SettingsConfigDict(env_prefix="RHP_LOGGING__", env_nested_delimiter="__")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO", description="Logging level")
    console: bool = Field(default=True, description="Enable console logging")
    file: bool = Field(default=True, description="Enable file logging")
    retention_days: int = Field(default=30, gt=0, description="Log file retention in days")


class AppConfig(BaseSettings):
    """Main application configuration combining all sections."""

    model_config = SettingsConfigDict(env_prefix="RHP_", env_nested_delimiter="__")

    # Configuration sections
    paths: PathsConfig = Field(default_factory=PathsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    model_config = SettingsConfigDict(
        env_prefix="RHP_",
        env_nested_delimiter="__",
        case_sensitive=False,
        env_file="config.yaml",
        env_file_encoding="utf-8",
    )

    @classmethod
    def load_from_yaml(cls, config_path: str | None = None) -> "AppConfig":
        """Load configuration from YAML file with environment overrides.

        Args:
            config_path: Path to YAML config file. Defaults to 'config.yaml'

        Returns:
            AppConfig instance with loaded configuration
        """
        from pathlib import Path

        import yaml

        # Determine config file path
        if config_path is None:
            config_path = "config.yaml"

        config_file = Path(config_path)

        # Load YAML if file exists
        yaml_data = {}
        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}

        # Create nested config objects
        config_data = {}
        for section_name in ["paths", "llm", "ingestion", "agents", "reporting", "logging"]:
            if section_name in yaml_data:
                config_data[section_name] = yaml_data[section_name]

        # Create AppConfig instance (environment variables will override automatically)
        return cls(**config_data)

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.paths.input_dir,
            self.paths.output_dir,
            self.paths.logs_dir,
            self.paths.data_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_log_file_path(self) -> Path:
        """Get the current log file path with date."""
        from datetime import datetime

        date_str = datetime.now().strftime("%Y-%m-%d")
        return self.paths.logs_dir / f"rhp-analyzer-{date_str}.log"

    def get_error_log_path(self) -> Path:
        """Get the error log file path."""
        return self.paths.logs_dir / "errors.log"
