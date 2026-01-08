"""Tests for configuration loader with priority resolution."""

import os
from pathlib import Path

import pytest

from rhp_analyzer.config.loader import ConfigLoader, ConfigurationError, load_config, load_config_for_testing


class TestConfigLoader:
    """Test configuration loader functionality."""

    def test_load_default_config(self):
        """Test loading with defaults only."""
        config = load_config_for_testing()

        # Should have default values
        assert config.paths.input_dir == Path("./data/input").resolve()
        assert config.llm.provider == "huggingface"
        assert config.ingestion.chunk_size == 1000
        assert config.agents.max_revisions == 2
        assert config.logging.level == "INFO"

    def test_yaml_override(self):
        """Test YAML file overrides defaults."""
        yaml_content = """
        paths:
          input_dir: "./custom/input"
        llm:
          temperature: 0.5
        ingestion:
          chunk_size: 2000
        """

        config = load_config_for_testing(yaml_content=yaml_content)

        # Use resolve() to compare paths correctly on Windows
        assert config.paths.input_dir == Path("./custom/input").resolve()
        assert config.llm.temperature == 0.5
        assert config.ingestion.chunk_size == 2000
        # Non-overridden values should remain defaults
        assert config.llm.provider == "huggingface"

    def test_env_var_override(self):
        """Test environment variables override YAML."""
        yaml_content = """
        llm:
          temperature: 0.5
        ingestion:
          chunk_size: 2000
        """

        env_vars = {
            "RHP_LLM__TEMPERATURE": "0.3",
            "RHP_INGESTION__BATCH_SIZE": "64",
        }

        config = load_config_for_testing(yaml_content=yaml_content, env_vars=env_vars)

        # Env var should override YAML
        assert config.llm.temperature == 0.3
        # Non-env var should use YAML value
        assert config.ingestion.chunk_size == 2000
        # New env var should be applied
        assert config.ingestion.batch_size == 64

    def test_cli_override(self):
        """Test CLI arguments override both YAML and env vars."""
        yaml_content = """
        llm:
          temperature: 0.5
        ingestion:
          chunk_size: 2000
        """

        env_vars = {
            "RHP_LLM__TEMPERATURE": "0.3",
        }

        cli_overrides = {"llm": {"temperature": 0.1}, "agents": {"max_revisions": 5}}

        config = load_config_for_testing(yaml_content=yaml_content, env_vars=env_vars, cli_overrides=cli_overrides)

        # CLI should override both YAML and env var
        assert config.llm.temperature == 0.1
        # CLI override for new value
        assert config.agents.max_revisions == 5
        # Non-overridden values should use YAML
        assert config.ingestion.chunk_size == 2000

    def test_env_var_parsing(self):
        """Test parsing of different environment variable types."""
        env_vars = {
            "RHP_AGENTS__PARALLEL_EXECUTION": "false",
            "RHP_AGENTS__MAX_REVISIONS": "3",
            "RHP_LLM__TIMEOUT": "60",  # timeout is int in schema
        }

        config = load_config_for_testing(env_vars=env_vars)

        assert config.agents.parallel_execution is False  # boolean
        assert config.agents.max_revisions == 3  # integer
        assert config.llm.timeout == 60  # integer (schema defines it as int)

    def test_invalid_yaml_error(self):
        """Test handling of invalid YAML file."""
        yaml_content = """
        invalid: yaml: content:
        - missing
          proper: structure
        """

        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            load_config_for_testing(yaml_content=yaml_content)

    def test_validation_error(self):
        """Test helpful validation error messages."""
        yaml_content = """
        llm:
          temperature: 5.0  # Invalid - too high
        ingestion:
          chunk_size: -100  # Invalid - negative
        """

        with pytest.raises(ConfigurationError) as exc_info:
            load_config_for_testing(yaml_content=yaml_content)

        error_msg = str(exc_info.value)
        assert "Configuration validation failed" in error_msg
        assert "llm -> temperature" in error_msg
        assert "ingestion -> chunk_size" in error_msg

    def test_missing_yaml_file(self):
        """Test handling of missing YAML file."""
        # Should work fine with missing YAML - use defaults + env vars
        config = load_config(config_path="/nonexistent/config.yaml")
        assert config.llm.provider == "huggingface"  # default value

    def test_config_sources_info(self):
        """Test configuration sources information."""
        loader = ConfigLoader("config.yaml")
        info = loader.get_config_sources_info()

        assert "yaml_file" in info
        assert "environment_variables" in info
        assert "cli_overrides" in info
        assert "path" in info["yaml_file"]
        assert "count" in info["environment_variables"]


def test_integration_with_existing_config():
    """Integration test with actual config.yaml file."""
    # This test assumes the config.yaml exists in the project root
    config_path = Path(__file__).parent.parent.parent.parent / "config.yaml"

    if config_path.exists():
        # Test loading existing config
        config = load_config(str(config_path))

        # Verify expected structure
        assert hasattr(config, "paths")
        assert hasattr(config, "llm")
        assert hasattr(config, "ingestion")
        assert hasattr(config, "agents")
        assert hasattr(config, "reporting")
        assert hasattr(config, "logging")

        # Test env var override
        old_temp = os.environ.get("RHP_LLM__TEMPERATURE")
        try:
            os.environ["RHP_LLM__TEMPERATURE"] = "0.9"
            config_override = load_config(str(config_path))
            assert config_override.llm.temperature == 0.9
        finally:
            if old_temp is None:
                os.environ.pop("RHP_LLM__TEMPERATURE", None)
            else:
                os.environ["RHP_LLM__TEMPERATURE"] = old_temp


if __name__ == "__main__":
    # Run a quick test
    print("Testing configuration loader...")

    # Test basic loading
    config = load_config_for_testing()
    print(f"✓ Default config loaded: {config.llm.provider}")

    # Test YAML override
    yaml_test = """
    llm:
      temperature: 0.8
    """
    config = load_config_for_testing(yaml_content=yaml_test)
    print(f"✓ YAML override: temperature = {config.llm.temperature}")

    # Test env var override
    config = load_config_for_testing(yaml_content=yaml_test, env_vars={"RHP_LLM__TEMPERATURE": "0.2"})
    print(f"✓ Env var override: temperature = {config.llm.temperature}")

    # Test CLI override
    config = load_config_for_testing(
        yaml_content=yaml_test, env_vars={"RHP_LLM__TEMPERATURE": "0.2"}, cli_overrides={"llm": {"temperature": 0.1}}
    )
    print(f"✓ CLI override: temperature = {config.llm.temperature}")

    print("All tests passed! ✓")
