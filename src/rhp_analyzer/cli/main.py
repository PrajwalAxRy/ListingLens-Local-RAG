"""
Main CLI application using Typer.
Provides entry point and command routing for RHP Analyzer.
"""
from pathlib import Path

import typer
from rich.console import Console

from rhp_analyzer.cli.commands.analyze import analyze_cmd
from rhp_analyzer.cli.commands.config import config_cmd
from rhp_analyzer.cli.commands.validate import validate_cmd
from rhp_analyzer.config.loader import ConfigLoader
from rhp_analyzer.utils.log_setup import setup_logging

# Initialize Typer app
app = typer.Typer(
    name="rhp-analyzer",
    help="RHP Analyzer - AI-powered analysis of Indian IPO Red Herring Prospectus documents",
    add_completion=False,
    rich_markup_mode="rich",
)

# Initialize Rich console for output
console = Console()

# Global state for configuration
_config: ConfigLoader | None = None


def version_callback(value: bool):
    """Display version information."""
    if value:
        try:
            from importlib.metadata import version

            app_version = version("rhp-analyzer")
        except Exception:
            # Fallback if package not installed
            app_version = "0.1.0"

        console.print(f"[bold cyan]RHP Analyzer[/bold cyan] version [green]{app_version}[/green]")
        raise typer.Exit()


# Module-level typer options
VERSION_OPTION = typer.Option(
    None,
    "--version",
    "-v",
    callback=version_callback,
    is_eager=True,
    help="Show version and exit.",
)
CONFIG_OPTION = typer.Option(
    None,
    "--config",
    "-c",
    help="Path to configuration file (default: ./config.yaml)",
    exists=True,
    dir_okay=False,
    readable=True,
)
VERBOSE_OPTION = typer.Option(
    False,
    "--verbose",
    help="Enable verbose output (DEBUG level logging)",
)


def load_config(config_path: Path | None = None) -> ConfigLoader:
    """
    Load configuration with proper precedence.
    Priority: Environment Variables > CLI config file > default config.yaml > defaults
    """
    global _config

    if _config is not None:
        return _config

    try:
        _config = ConfigLoader(config_path=str(config_path)) if config_path else ConfigLoader()

        return _config

    except Exception as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e!s}", style="red")
        console.print(
            "\n[yellow]Tip:[/yellow] Check your config.yaml file or use --config to specify a different path."
        )
        raise typer.Exit(code=1) from e


@app.callback()
def main(
    ctx: typer.Context,
    version: bool | None = VERSION_OPTION,
    config: Path | None = CONFIG_OPTION,
    verbose: bool = VERBOSE_OPTION,
):
    """
    RHP Analyzer - Comprehensive analysis of Indian IPO prospectus documents.

    Use [bold cyan]rhp-analyzer COMMAND --help[/bold cyan] for command-specific help.
    """
    # Store config path in context for subcommands
    ctx.obj = {
        "config_path": config,
        "verbose": verbose,
    }

    # Initialize logging early
    try:
        log_level = "DEBUG" if verbose else "INFO"
        setup_logging(log_level=log_level)
    except Exception as e:
        console.print(f"[bold red]Logging Setup Error:[/bold red] {e!s}", style="red")
        raise typer.Exit(code=1) from e


app.command(name="analyze")(analyze_cmd)
app.command(name="validate")(validate_cmd)
app.command(name="config")(config_cmd)


if __name__ == "__main__":
    app()
