"""
Analyze command - Main RHP analysis workflow.
"""
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console

console = Console()

# Module-level defaults for typer arguments
PDF_ARGUMENT = typer.Argument(
    ...,
    help="Path to RHP PDF file",
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
)
OUTPUT_OPTION = typer.Option(
    None,
    "--output-dir",
    "-o",
    help="Output directory for analysis results (default: from config)",
)
DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    help="Validate input without performing actual analysis",
)
VERBOSE_OPTION = typer.Option(
    False,
    "--verbose",
    "-v",
    help="Enable verbose (debug) logging output",
)


def validate_pdf_path(pdf_path: Path) -> None:
    """
    Validate PDF file path and properties.

    Args:
        pdf_path: Path to PDF file

    Raises:
        typer.BadParameter: If validation fails
    """
    # Check if file exists
    if not pdf_path.exists():
        raise typer.BadParameter(f"File not found: {pdf_path}")

    # Check if it's a file (not directory)
    if not pdf_path.is_file():
        raise typer.BadParameter(f"Path is not a file: {pdf_path}")

    # Check if readable
    if not pdf_path.stat().st_mode & 0o444:
        raise typer.BadParameter(f"File is not readable: {pdf_path}")

    # Check file extension
    if pdf_path.suffix.lower() != ".pdf":
        raise typer.BadParameter(f"File must be a PDF (got {pdf_path.suffix})")

    # Check file size (basic sanity check)
    size_bytes = pdf_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    # Very minimal PDFs for testing can be < 1KB, but < 10 bytes is definitely wrong
    if size_bytes < 10:
        raise typer.BadParameter(f"PDF file too small ({size_bytes} bytes) - may be corrupted")
    if size_mb > 500:
        console.print(f"[yellow]⚠[/yellow] Large PDF file ({size_mb:.1f} MB) - processing may take longer")


def analyze_cmd(
    ctx: typer.Context,
    pdf_path: Path = PDF_ARGUMENT,
    output_dir: Path | None = OUTPUT_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    verbose: bool = VERBOSE_OPTION,
):
    """
    Analyze an RHP document and generate comprehensive investment report.

    This command processes the PDF through the complete analysis pipeline:
    ingestion → analysis → report generation.

    Example:
        rhp-analyzer analyze path/to/rhp.pdf
        rhp-analyzer analyze rhp.pdf --output-dir ./reports --dry-run
    """
    console.print("\n[bold cyan]RHP Analyzer - Analyze Command[/bold cyan]\n")

    # Setup logging level based on verbose flag
    if verbose:
        logger.remove()  # Remove default handler
        logger.add(
            lambda msg: console.print(msg, end=""),
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level="DEBUG",
        )
        logger.debug("Verbose logging enabled")
    else:
        logger.remove()
        logger.add(
            lambda msg: console.print(msg, end=""),
            format="<level>{level: <8}</level> | <level>{message}</level>",
            level="INFO",
        )

    # Get config from context
    config_path = ctx.obj.get("config_path")

    # Validate PDF path
    try:
        validate_pdf_path(pdf_path)
        logger.info(f"PDF validation passed: {pdf_path}")
    except typer.BadParameter as exc:
        console.print(f"[red]✗[/red] Validation error: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]✓[/green] PDF Path: {pdf_path}")
    console.print(f"[green]✓[/green] Output Directory: {output_dir or '[from config]'}")
    console.print(f"[green]✓[/green] Dry Run: {dry_run}")
    console.print(f"[green]✓[/green] Config: {config_path or '[default]'}")
    console.print(f"[green]✓[/green] Verbose: {verbose}")

    if dry_run:
        console.print("\n[yellow]⚠[/yellow] [bold]Dry run mode[/bold] - validation only, no processing performed.")
        console.print("[green]✓[/green] PDF file is readable and exists.")
        console.print("[green]✓[/green] All validation checks passed.")
        return

    console.print("\n[yellow]i[/yellow] [bold]Analysis pipeline not yet implemented.[/bold]")
    console.print("[dim]This is a placeholder. Full implementation coming in Phase 2+.[/dim]")

    # Log the 6-phase pipeline structure (for Phase 2+)
    logger.debug("Pipeline phases:")
    logger.debug("  Phase 1: Ingestion & Preprocessing")
    logger.debug("  Phase 2: Chunking & Embedding")
    logger.debug("  Phase 3: Agent Analysis")
    logger.debug("  Phase 4: Critique & Verification")
    logger.debug("  Phase 5: Report Generation")
    logger.debug("  Phase 6: Output & Cleanup")

    # TODO: Phase 2+ Implementation
    # 1. Load configuration
    # 2. Initialize ingestion pipeline
    # 3. Process PDF through agents
    # 4. Generate report
    # 5. Save outputs


def analyze(
    pdf_path: Path,
    output_dir: Path | None = None,
    config_path: Path | None = None,
    *,
    dry_run: bool = False,
    verbose: bool = False,
) -> int | None:
    """
    Analyze an RHP document and generate comprehensive investment report.

    This command processes the PDF through the complete analysis pipeline:
    ingestion → analysis → report generation.

    Args:
        pdf_path: Path to RHP PDF file
        output_dir: Output directory for analysis results (default: from config)
        config_path: Path to configuration file (optional)
        dry_run: Validate input without performing actual analysis
        verbose: Enable verbose (debug) logging output

    Returns:
        Exit code (0 for success, 1 for failure) or None
    """
    console.print("\n[bold cyan]RHP Analyzer - Analyze Command[/bold cyan]\n")

    # Setup logging level based on verbose flag
    if verbose:
        logger.remove()  # Remove default handler
        logger.add(
            lambda msg: console.print(msg, end=""),
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            level="DEBUG",
        )
        logger.debug("Verbose logging enabled")
    else:
        logger.remove()
        logger.add(
            lambda msg: console.print(msg, end=""),
            format="<level>{level: <8}</level> | <level>{message}</level>",
            level="INFO",
        )

    # Validate PDF path
    try:
        validate_pdf_path(pdf_path)
        logger.info(f"PDF validation passed: {pdf_path}")
    except typer.BadParameter as exc:
        console.print(f"[red]✗[/red] Validation error: {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]✓[/green] PDF Path: {pdf_path}")
    console.print(f"[green]✓[/green] Output Directory: {output_dir or '[from config]'}")
    console.print(f"[green]✓[/green] Dry Run: {dry_run}")
    console.print(f"[green]✓[/green] Config: {config_path or '[default]'}")
    console.print(f"[green]✓[/green] Verbose: {verbose}")

    if dry_run:
        console.print("\n[yellow]⚠[/yellow] [bold]Dry run mode[/bold] - validation only, no processing performed.")
        console.print("[green]✓[/green] PDF file is readable and exists.")
        console.print("[green]✓[/green] All validation checks passed.")
        return

    console.print("\n[yellow]i[/yellow] [bold]Analysis pipeline not yet implemented.[/bold]")
    console.print("[dim]This is a placeholder. Full implementation coming in Phase 2+.[/dim]")

    # Log the 6-phase pipeline structure (for Phase 2+)
    logger.debug("Pipeline phases:")
    logger.debug("  Phase 1: Ingestion & Preprocessing")
    logger.debug("  Phase 2: Chunking & Embedding")
    logger.debug("  Phase 3: Agent Analysis")
    logger.debug("  Phase 4: Critique & Verification")
    logger.debug("  Phase 5: Report Generation")
    logger.debug("  Phase 6: Output & Cleanup")

    # TODO: Phase 2+ Implementation
    # 1. Load configuration
    # 2. Initialize ingestion pipeline
    # 3. Process PDF through agents
    # 4. Generate report
    # 5. Save outputs

    return 0
