"""
Config command - Display and validate configuration.
"""
import typer
from rich.console import Console
from rich.table import Table

from rhp_analyzer.config.loader import ConfigLoader

console = Console()


def config_cmd(
    ctx: typer.Context,
    show: bool = typer.Option(
        True,
        "--show",
        help="Show current configuration",
    ),
):
    """
    Display current configuration settings.

    Shows the merged configuration from all sources:
    - Environment variables (highest priority)
    - Config file from --config option
    - Default config.yaml
    - Built-in defaults (lowest priority)

    Example:
        rhp-analyzer config
        rhp-analyzer config --config custom-config.yaml
    """
    console.print("\n[bold cyan]RHP Analyzer - Configuration[/bold cyan]\n")

    # Get config path from context
    config_path = ctx.obj.get("config_path")

    try:
        # Load configuration
        if config_path:
            loader = ConfigLoader(config_path=str(config_path))
            console.print(f"[green]✓[/green] Using config file: [bold]{config_path}[/bold]")
        else:
            loader = ConfigLoader()
            console.print("[green]✓[/green] Using default config: [bold]config.yaml[/bold]")

        # Actually load the configuration
        config = loader.load_config()
        console.print("[green]✓[/green] Configuration loaded successfully\n")

        if show:
            # Display paths configuration
            paths_table = Table(title="Paths Configuration", show_header=True, header_style="bold cyan")
            paths_table.add_column("Setting", style="dim")
            paths_table.add_column("Value", style="green")

            paths_table.add_row("Input Directory", str(config.paths.input_dir))
            paths_table.add_row("Output Directory", str(config.paths.output_dir))
            paths_table.add_row("Logs Directory", str(config.paths.logs_dir))
            paths_table.add_row("Data Directory", str(config.paths.data_dir))

            console.print(paths_table)
            console.print()

            # Display LLM configuration
            llm_table = Table(title="LLM Configuration", show_header=True, header_style="bold cyan")
            llm_table.add_column("Setting", style="dim")
            llm_table.add_column("Value", style="green")

            llm_table.add_row("Provider", config.llm.provider)
            llm_table.add_row("Context Model", config.llm.context_model)
            llm_table.add_row("Reasoning Model", config.llm.reasoning_model)
            llm_table.add_row("Temperature", str(config.llm.temperature))
            llm_table.add_row("Max Tokens", str(config.llm.max_tokens))
            llm_table.add_row("Timeout", f"{config.llm.timeout}s")

            console.print(llm_table)
            console.print()

            # Display ingestion configuration
            ingestion_table = Table(title="Ingestion Configuration", show_header=True, header_style="bold cyan")
            ingestion_table.add_column("Setting", style="dim")
            ingestion_table.add_column("Value", style="green")

            ingestion_table.add_row("Chunk Size", str(config.ingestion.chunk_size))
            ingestion_table.add_row("Chunk Overlap", str(config.ingestion.chunk_overlap))
            ingestion_table.add_row("Min Chunk Size", str(config.ingestion.min_chunk_size))
            ingestion_table.add_row("Batch Size", str(config.ingestion.batch_size))

            console.print(ingestion_table)
            console.print()

            # Display enabled agents
            console.print("[bold cyan]Enabled Agents:[/bold cyan]")
            for agent in config.agents.enabled:
                console.print(f"  [green]✓[/green] {agent}")
            console.print()

            # Display reporting configuration
            console.print("[bold cyan]Reporting Configuration:[/bold cyan]")
            console.print(f"  Formats: {', '.join(config.reporting.formats)}")
            console.print(f"  Template: {config.reporting.template}")
            console.print(f"  Include Appendices: {config.reporting.include_appendices}")
            console.print()

    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Configuration Error: {e!s}", style="red")
        console.print("\n[yellow]Tip:[/yellow] Check your config.yaml file syntax and required fields.")
        raise typer.Exit(code=1) from e
