"""Test configuration schema with validation and environment variables."""

import os
import tempfile
from pathlib import Path

import pytest
from rhp_analyzer.config import AppConfig


def test_default_configuration():
    """Test that default configuration loads correctly."""
    config = AppConfig()

    # Test default values
    assert config.paths.input_dir == Path("./data/input").resolve()
    assert config.llm.provider == "huggingface"
    assert config.llm.temperature == 0.1
    assert config.ingestion.chunk_size == 1000
    assert config.agents.parallel_execution is True
    assert config.reporting.formats == ["markdown", "pdf"]
    assert config.logging.level == "INFO"


def test_yaml_configuration_loading():
    """Test loading configuration from YAML file."""
    yaml_content = """
paths:
  input_dir: "./custom/input"
  output_dir: "./custom/output"

llm:
  temperature: 0.5
  max_tokens: 2048

ingestion:
  chunk_size: 500
  batch_size: 16

agents:
  enabled:
    - architect
    - forensic
  max_revisions: 3

reporting:
  formats:
    - markdown
  template: custom

logging:
  level: DEBUG
  retention_days: 15
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        f.flush()
        temp_file_path = f.name

    try:
        config = AppConfig.load_from_yaml(temp_file_path)

        # Test loaded values
        assert config.paths.input_dir.name == "input"
        assert "custom" in str(config.paths.input_dir)
        assert config.llm.temperature == 0.5
        assert config.llm.max_tokens == 2048
        assert config.ingestion.chunk_size == 500
        assert config.ingestion.batch_size == 16
        assert config.agents.enabled == ["architect", "forensic"]
        assert config.agents.max_revisions == 3
        assert config.reporting.formats == ["markdown"]
        assert config.reporting.template == "custom"
        assert config.logging.level == "DEBUG"
        assert config.logging.retention_days == 15

    finally:
        # Windows-safe cleanup
        import contextlib

        with contextlib.suppress(PermissionError):
            os.unlink(temp_file_path)
        # On Windows, might need to wait a moment if first attempt fails
        import time

        time.sleep(0.1)
        with contextlib.suppress(PermissionError):
            os.unlink(temp_file_path)


def test_environment_variable_overrides():
    """Test environment variable overrides using BaseSettings."""
    from unittest.mock import patch

    # Set environment variables
    env_vars = {
        "RHP_PATHS__INPUT_DIR": "/tmp/test_input",
        "RHP_PATHS__OUTPUT_DIR": "/tmp/test_output",
        "RHP_LLM__TEMPERATURE": "0.7",
        "RHP_LLM__MAX_TOKENS": "8192",
        "RHP_INGESTION__CHUNK_SIZE": "2000",
        "RHP_AGENTS__MAX_REVISIONS": "5",
    }

    with patch.dict(os.environ, env_vars, clear=False):
        config = AppConfig()

        # Check that environment variables override defaults
        # Use endswith to handle Windows vs Unix path differences
        assert str(config.paths.input_dir).endswith("test_input")
        assert str(config.paths.output_dir).endswith("test_output")
        assert config.llm.temperature == 0.7
        assert config.llm.max_tokens == 8192
        assert config.ingestion.chunk_size == 2000
        assert config.agents.max_revisions == 5


def test_validation_rules():
    """Test that validation rules are enforced."""
    # Test invalid temperature
    with pytest.raises(ValueError):
        AppConfig(llm={"temperature": 3.0})  # Should be <= 2.0

    # Test invalid chunk overlap
    with pytest.raises(ValueError):
        AppConfig(ingestion={"chunk_size": 100, "chunk_overlap": 150})

    # Test invalid agent name
    with pytest.raises(ValueError):
        AppConfig(agents={"enabled": ["invalid_agent"]})


def test_directory_creation():
    """Test that ensure_directories creates necessary directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = AppConfig(
            paths={
                "input_dir": temp_path / "input",
                "output_dir": temp_path / "output",
                "logs_dir": temp_path / "logs",
                "data_dir": temp_path / "data",
            }
        )

        # Directories shouldn't exist yet
        assert not config.paths.input_dir.exists()

        # Create directories
        config.ensure_directories()

        # Now they should exist
        assert config.paths.input_dir.exists()
        assert config.paths.output_dir.exists()
        assert config.paths.logs_dir.exists()
        assert config.paths.data_dir.exists()


def test_log_file_paths():
    """Test log file path generation."""
    config = AppConfig()

    log_path = config.get_log_file_path()
    error_path = config.get_error_log_path()

    assert str(log_path).endswith(".log")
    assert "rhp-analyzer" in str(log_path)
    assert str(error_path).endswith("errors.log")


if __name__ == "__main__":
    # Run basic tests
    test_default_configuration()
    test_validation_rules()
    print("Configuration schema tests passed!")
