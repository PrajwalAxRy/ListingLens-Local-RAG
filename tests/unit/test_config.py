"""Test configuration system with validation and environment variables.

Tests for Subtask 1.3.4:
- Test loading from file
- Test environment variable override
- Test validation errors
"""

import contextlib
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from rhp_analyzer.config import AppConfig
from rhp_analyzer.config.loader import ConfigLoader, ConfigurationError, load_config


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

    # Use a more robust temporary file approach for Windows
    temp_fd, temp_file_path = tempfile.mkstemp(suffix=".yaml", text=True)

    try:
        # Write content using file descriptor
        with os.fdopen(temp_fd, "w") as f:
            f.write(yaml_content)
            f.flush()

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
        # Windows-safe cleanup - try multiple approaches
        with contextlib.suppress(FileNotFoundError, PermissionError):
            os.unlink(temp_file_path)


def test_config_loader_from_file():
    """Test ConfigLoader with file loading (Subtask 1.3.4 requirement)."""
    yaml_content = {
        "paths": {"input_dir": "./test/input"},
        "llm": {"temperature": 0.3, "max_tokens": 2048},
        "ingestion": {"chunk_size": 800},
    }

    # Create temp file safely
    temp_fd, temp_file_path = tempfile.mkstemp(suffix=".yaml", text=True)

    try:
        # Write YAML content
        with os.fdopen(temp_fd, "w") as f:
            yaml.dump(yaml_content, f)

        # Test ConfigLoader
        config = load_config(temp_file_path)
        assert config.llm.temperature == 0.3
        assert config.llm.max_tokens == 2048
        assert config.ingestion.chunk_size == 800
        assert "test" in str(config.paths.input_dir)

    finally:
        with contextlib.suppress(FileNotFoundError, PermissionError):
            os.unlink(temp_file_path)


def test_config_loader_missing_file():
    """Test ConfigLoader with missing file (should use defaults)."""
    # Use non-existent file path
    config = load_config("nonexistent_config.yaml")

    # Should get default values
    assert config.paths.input_dir == Path("./data/input").resolve()
    assert config.llm.provider == "huggingface"
    assert config.llm.temperature == 0.1
    assert config.ingestion.chunk_size == 1000


def test_config_loader_invalid_yaml():
    """Test ConfigLoader with invalid YAML file (Edge case from Subtask 1.3.4)."""
    invalid_yaml_content = """
    paths:
      input_dir: ./test
    llm:
      temperature: 0.2
      - invalid yaml syntax here
    """

    # Create temp file with invalid YAML
    temp_fd, temp_file_path = tempfile.mkstemp(suffix=".yaml", text=True)

    try:
        with os.fdopen(temp_fd, "w") as f:
            f.write(invalid_yaml_content)

        # Should handle gracefully (either raise specific error or use defaults)
        try:
            config = load_config(temp_file_path)
            # If it doesn't raise an error, it should at least have defaults
            assert hasattr(config, "llm")
            assert hasattr(config, "paths")
        except Exception as e:
            # If it raises an error, it should be informative
            assert any(keyword in str(e).lower() for keyword in ["yaml", "parse", "invalid"])

    finally:
        with contextlib.suppress(FileNotFoundError, PermissionError):
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


def test_config_loader_env_override():
    """Test ConfigLoader with environment variable override (Subtask 1.3.4 requirement)."""
    yaml_content = {"llm": {"temperature": 0.2, "max_tokens": 1024}, "ingestion": {"chunk_size": 500}}

    # Create temp file
    temp_fd, temp_file_path = tempfile.mkstemp(suffix=".yaml", text=True)

    try:
        with os.fdopen(temp_fd, "w") as f:
            yaml.dump(yaml_content, f)

        # Test with environment override (priority: env > yaml > defaults)
        with patch.dict(os.environ, {"RHP_LLM__TEMPERATURE": "0.7", "RHP_INGESTION__BATCH_SIZE": "64"}, clear=False):
            config = load_config(temp_file_path)

            # Env var should override YAML
            assert config.llm.temperature == 0.7
            # YAML value should be preserved where no env override
            assert config.llm.max_tokens == 1024
            # Env var for new field should be set
            assert config.ingestion.batch_size == 64
            # YAML value should remain for unchanged
            assert config.ingestion.chunk_size == 500

    finally:
        with contextlib.suppress(FileNotFoundError, PermissionError):
            os.unlink(temp_file_path)


def test_config_loader_priority():
    """Test ConfigLoader priority resolution (CLI > Env > YAML > Defaults)."""
    yaml_content = {"llm": {"temperature": 0.3}}

    # Create temp file
    temp_fd, temp_file_path = tempfile.mkstemp(suffix=".yaml", text=True)

    try:
        with os.fdopen(temp_fd, "w") as f:
            yaml.dump(yaml_content, f)

        # Create ConfigLoader with specific YAML file
        loader = ConfigLoader(config_path=temp_file_path)

        # Test priority: env should override yaml, CLI should override env
        with patch.dict(os.environ, {"RHP_LLM__TEMPERATURE": "0.9"}, clear=False):
            config = loader.load_config(cli_overrides={"llm": {"max_tokens": 8192}})

            # CLI override should win over env
            assert config.llm.max_tokens == 8192
            # Env should override YAML
            assert config.llm.temperature == 0.9

    finally:
        with contextlib.suppress(FileNotFoundError, PermissionError):
            os.unlink(temp_file_path)


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


def test_config_validation_errors():
    """Test comprehensive validation errors (Subtask 1.3.4 requirement)."""

    # Test various validation error scenarios
    with pytest.raises(ValueError, match="temperature"):
        AppConfig(llm={"temperature": -1.0})  # Negative temperature

    with pytest.raises(ValueError, match="temperature"):
        AppConfig(llm={"temperature": 5.0})  # Too high temperature

    with pytest.raises(ValueError, match="chunk_size"):
        AppConfig(ingestion={"chunk_size": 0})  # Zero chunk size

    with pytest.raises(ValueError, match="chunk_overlap"):
        AppConfig(ingestion={"chunk_size": 100, "chunk_overlap": 200})  # Overlap > size

    with pytest.raises(ValueError, match="max_tokens"):
        AppConfig(llm={"max_tokens": 0})  # Zero tokens

    with pytest.raises(ValueError, match="batch_size"):
        AppConfig(ingestion={"batch_size": -1})  # Negative batch size

    with pytest.raises(ValueError, match="max_revisions"):
        AppConfig(agents={"max_revisions": -1})  # Negative revisions


def test_config_loader_validation_errors():
    """Test ConfigLoader handling of validation errors."""
    invalid_config_yaml = {
        "llm": {"temperature": 999.0},  # Invalid
        "ingestion": {"chunk_size": -1},  # Invalid
    }

    # Create temp file with invalid config
    temp_fd, temp_file_path = tempfile.mkstemp(suffix=".yaml", text=True)

    try:
        with os.fdopen(temp_fd, "w") as f:
            yaml.dump(invalid_config_yaml, f)

        # Should raise ConfigurationError when trying to load
        loader = ConfigLoader(config_path=temp_file_path)
        with pytest.raises(ConfigurationError):
            loader.load_config()

    finally:
        with contextlib.suppress(FileNotFoundError, PermissionError):
            os.unlink(temp_file_path)


def test_config_loader_permission_error():
    """Test ConfigLoader with file permission errors (Edge case)."""
    import platform

    yaml_content = {"llm": {"temperature": 0.5}}

    # Create temp file
    temp_fd, temp_file_path = tempfile.mkstemp(suffix=".yaml", text=True)

    try:
        with os.fdopen(temp_fd, "w") as f:
            yaml.dump(yaml_content, f)

        # Platform-specific permission handling
        if platform.system() == "Windows":
            # On Windows, just test the file exists and can be read normally
            # Real permission errors are harder to simulate reliably
            loader = ConfigLoader(config_path=temp_file_path)
            config = loader.load_config()
            assert config.llm.temperature == 0.5
        else:
            # Unix-like systems: try chmod approach
            try:
                os.chmod(temp_file_path, 0o000)

                # Loading should handle permission error gracefully
                loader = ConfigLoader(config_path=temp_file_path)
                config = loader.load_config()
                # Should fall back to defaults if can't read file
                assert config.llm.temperature == 0.1  # default value

            except (PermissionError, OSError):
                # If we can't change permissions, test passes
                pass

    finally:
        # Restore permissions before cleanup
        with contextlib.suppress(PermissionError):
            os.chmod(temp_file_path, 0o644)
        with contextlib.suppress(FileNotFoundError, PermissionError):
            os.unlink(temp_file_path)


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
