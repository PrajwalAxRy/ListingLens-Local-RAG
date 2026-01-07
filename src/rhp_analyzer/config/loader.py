"""Configuration loader with priority-based resolution.

This module implements configuration loading with the following priority order:
1. CLI arguments (highest priority)
2. Environment variables (RHP_ prefix)
3. YAML configuration file
4. Default values (lowest priority)
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .schema import AppConfig


class ConfigurationError(Exception):
    """Raised when configuration loading fails."""

    pass


class ConfigLoader:
    """Configuration loader with priority-based resolution.

    Handles loading configuration from multiple sources with proper precedence:
    CLI arguments > Environment variables > YAML file > Defaults
    """

    def __init__(self, config_path: str | None = None):
        """Initialize configuration loader.

        Args:
            config_path: Path to YAML configuration file.
                        Defaults to 'config.yaml' in current directory.
        """
        self.config_path = config_path or "config.yaml"
        self._yaml_data: dict[str, Any] = {}
        self._cli_overrides: dict[str, Any] = {}

    def load_config(
        self,
        cli_overrides: dict[str, Any] | None = None,
        validate: bool = True,
    ) -> AppConfig:
        """Load configuration with full precedence resolution.

        Args:
            cli_overrides: Dictionary of CLI argument overrides
            validate: Whether to run validation on the final configuration

        Returns:
            AppConfig: Validated configuration object

        Raises:
            ConfigurationError: If configuration loading or validation fails
        """
        try:
            # Store CLI overrides for precedence resolution
            self._cli_overrides = cli_overrides or {}

            # Step 1: Load YAML configuration (if exists)
            self._load_yaml_config()

            # Step 2: Create merged configuration dict with precedence
            merged_config = self._merge_all_sources()

            # Step 3: Create AppConfig
            # Always use normal instantiation to ensure nested models are properly created
            # The 'validate' parameter now only controls whether we skip directory creation
            # (useful for testing without side effects)
            config = AppConfig(**merged_config)

            # Step 4: Ensure required directories exist (skip in non-validation mode for testing)
            if validate:
                config.ensure_directories()

            return config

        except ValidationError as e:
            self._raise_helpful_error(e)
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}") from e

    def _load_yaml_config(self) -> None:
        """Load YAML configuration file if it exists."""
        config_file = Path(self.config_path)

        if not config_file.exists():
            # YAML file is optional - use defaults + env vars
            self._yaml_data = {}
            return

        try:
            with open(config_file, encoding="utf-8") as f:
                self._yaml_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {config_file}: {e}") from e
        except Exception as e:
            raise ConfigurationError(f"Failed to read {config_file}: {e}") from e

    def _merge_all_sources(self) -> dict[str, Any]:
        """Merge configuration from all sources with proper precedence.

        Returns:
            Dict with merged configuration values
        """
        # Start with YAML data as base
        merged = self._yaml_data.copy()

        # Apply environment variable overrides
        env_overrides = self._get_env_overrides()
        self._deep_merge(merged, env_overrides)

        # Apply CLI overrides (highest priority)
        self._deep_merge(merged, self._cli_overrides)

        return merged

    def _get_env_overrides(self) -> dict[str, Any]:
        """Extract relevant environment variables with RHP_ prefix.

        Returns:
            Dict with nested structure for environment overrides
        """
        env_vars = {k: v for k, v in os.environ.items() if k.startswith("RHP_")}

        if not env_vars:
            return {}

        # Convert flat env vars to nested dict structure
        nested = {}

        for env_key, env_value in env_vars.items():
            # Remove RHP_ prefix
            key = env_key[4:]  # Remove "RHP_"

            # Split by double underscores for nesting
            parts = key.split("__")

            # Convert value to appropriate type
            parsed_value = self._parse_env_value(env_value)

            # Create nested structure
            current = nested
            for part in parts[:-1]:
                part = part.lower()
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set final value
            current[parts[-1].lower()] = parsed_value

        return nested

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate Python type.

        Args:
            value: String value from environment variable

        Returns:
            Parsed value (str, int, float, bool, or list)
        """
        if not value:
            return value

        # Handle boolean values
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        if value.lower() in ("false", "no", "0", "off"):
            return False

        # Handle lists (comma-separated)
        if "," in value:
            return [item.strip() for item in value.split(",") if item.strip()]

        # Handle numeric values
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        return value

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> None:
        """Deep merge override dict into base dict.

        Args:
            base: Base dictionary to merge into (modified in place)
            override: Override dictionary to merge from
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _raise_helpful_error(self, validation_error: ValidationError) -> None:
        """Convert Pydantic validation error to helpful configuration error.

        Args:
            validation_error: Pydantic validation error

        Raises:
            ConfigurationError: With helpful error message
        """
        error_messages = []

        for error in validation_error.errors():
            loc = " -> ".join(str(x) for x in error["loc"])
            msg = error["msg"]
            error_messages.append(f"  {loc}: {msg}")

        helpful_msg = (
            "Configuration validation failed:\n"
            + "\n".join(error_messages)
            + "\n\nPlease check your configuration in:\n"
            f"  1. {self.config_path} (YAML file)\n"
            "  2. Environment variables (RHP_* prefix)\n"
            "  3. CLI arguments\n"
        )

        raise ConfigurationError(helpful_msg) from validation_error

    def get_config_sources_info(self) -> dict[str, Any]:
        """Get information about configuration sources for debugging.

        Returns:
            Dict with information about config sources
        """
        config_file = Path(self.config_path)
        env_vars = {k: v for k, v in os.environ.items() if k.startswith("RHP_")}

        return {
            "yaml_file": {
                "path": str(config_file.absolute()),
                "exists": config_file.exists(),
                "readable": config_file.exists() and os.access(config_file, os.R_OK),
            },
            "environment_variables": {
                "count": len(env_vars),
                "variables": list(env_vars.keys()),
            },
            "cli_overrides": {
                "count": len(self._cli_overrides),
                "sections": list(self._cli_overrides.keys()),
            },
        }


def load_config(
    config_path: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
    validate: bool = True,
) -> AppConfig:
    """Convenience function to load configuration.

    Args:
        config_path: Path to YAML configuration file
        cli_overrides: Dictionary of CLI argument overrides
        validate: Whether to validate the configuration

    Returns:
        AppConfig: Loaded and validated configuration

    Raises:
        ConfigurationError: If configuration loading fails
    """
    loader = ConfigLoader(config_path)
    return loader.load_config(cli_overrides, validate)


def load_config_for_testing(
    yaml_content: str | None = None,
    env_vars: dict[str, str] | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load configuration for testing purposes.

    Args:
        yaml_content: YAML content as string
        env_vars: Environment variables to set temporarily
        cli_overrides: CLI override values

    Returns:
        AppConfig: Loaded configuration
    """
    import tempfile

    # Create temporary YAML file if content provided
    config_path = None
    if yaml_content:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            f.write(yaml_content)
            config_path = f.name

    # Set environment variables temporarily
    old_env = {}
    if env_vars:
        for key, value in env_vars.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value

    try:
        loader = ConfigLoader(config_path)
        return loader.load_config(cli_overrides, validate=False)
    finally:
        # Restore environment variables
        for key, old_value in old_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

        # Clean up temporary file
        if config_path:
            os.unlink(config_path)
