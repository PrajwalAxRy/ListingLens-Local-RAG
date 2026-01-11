"""
Progress display utilities for RHP Analyzer.

Provides progress bars and status updates for long-running operations
with support for both TTY and non-TTY environments.
"""
import sys
from contextlib import contextmanager

from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Initialize Rich console
console = Console()


class ProgressTracker:
    """
    Tracks progress for multi-phase operations.

    Supports both visual progress bars (for TTY) and text-based updates
    (for non-TTY environments like logs or CI/CD).
    """

    def __init__(
        self,
        total_phases: int = 6,
    ):
        """
        Initialize the progress tracker.

        Args:
            total_phases: Total number of phases in the pipeline
        """
        self.total_phases = total_phases
        self.current_phase = 0
        self.is_tty = sys.stdout.isatty()

        # Initialize Rich Progress with custom columns
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            disable=not self.is_tty,
        )
        self.overall_task: TaskID | None = None
        self.current_task: TaskID | None = None

    def start(self, description: str = "Processing RHP Document"):
        """
        Start the progress tracker.

        Args:
            description: Description of the overall operation
        """
        self.progress.start()
        self.overall_task = self.progress.add_task(
            f"[cyan]{description}",
            total=self.total_phases,
        )
        logger.info(f"Started: {description}")

    def start_phase(
        self,
        phase_name: str,
        total_steps: int | None = None,
    ):
        """
        Start a new phase of processing.

        Args:
            phase_name: Name of the phase (e.g., "PDF Ingestion")
            total_steps: Total number of steps in this phase
        """
        self.current_phase += 1

        if self.is_tty:
            # Remove previous phase task if exists
            if self.current_task is not None:
                self.progress.remove_task(self.current_task)

            # Add new phase task
            self.current_task = self.progress.add_task(
                f"[green]Phase {self.current_phase}/{self.total_phases}: {phase_name}",
                total=total_steps,
            )
        else:
            # Non-TTY: Just log the phase
            logger.info(f"Phase {self.current_phase}/{self.total_phases}: {phase_name}")

        # Update overall progress
        if self.overall_task is not None:
            self.progress.update(
                self.overall_task,
                completed=self.current_phase - 1,
            )

    def update(
        self,
        advance: int = 1,
        description: str | None = None,
    ):
        """
        Update progress for the current phase.

        Args:
            advance: Number of steps to advance
            description: Optional description update
        """
        if self.current_task is not None and self.is_tty:
            update_kwargs = {"advance": advance}
            if description:
                update_kwargs["description"] = (
                    f"[green]Phase {self.current_phase}/{self.total_phases}: " f"{description}"
                )
            self.progress.update(self.current_task, **update_kwargs)
        elif description:
            # Non-TTY: Log description updates
            logger.debug(description)

    def complete_phase(self):
        """Mark the current phase as complete."""
        if self.current_task is not None and self.is_tty:
            # Get task by ID - tasks is a list, iterate to find matching task
            task = None
            for t in self.progress.tasks:
                if t.id == self.current_task:
                    task = t
                    break

            if task is not None and task.total is not None:
                self.progress.update(
                    self.current_task,
                    completed=task.total,
                )

        # Update overall progress
        if self.overall_task is not None:
            self.progress.update(
                self.overall_task,
                completed=self.current_phase,
            )

        logger.info(f"Completed phase {self.current_phase}/{self.total_phases}")

    def stop(self):
        """Stop the progress tracker (alias for error handling)."""
        self.progress.stop()

    def finish(self, message: str = "Processing complete"):
        """
        Finish progress tracking.

        Args:
            message: Completion message to display
        """
        if self.overall_task is not None:
            self.progress.update(
                self.overall_task,
                completed=self.total_phases,
            )

        self.progress.stop()

        if self.is_tty:
            console.print(f"[bold green]✓[/bold green] {message}")
        else:
            logger.info(message)

    def error(self, message: str):
        """
        Display an error message and stop progress tracking.

        Args:
            message: Error message to display
        """
        self.progress.stop()

        if self.is_tty:
            console.print(f"[bold red]✗[/bold red] {message}")
        else:
            logger.error(message)


@contextmanager
def track_progress(
    description: str = "Processing",
    total_phases: int = 6,
):
    """
    Context manager for tracking progress of an operation.

    Usage:
        with track_progress("Analyzing RHP", total_phases=6) as tracker:
            tracker.start_phase("Ingestion", total_steps=100)
            for i in range(100):
                # Do work
                tracker.update(advance=1)
            tracker.complete_phase()

    Args:
        description: Description of the operation
        total_phases: Total number of phases

    Yields:
        ProgressTracker: The progress tracker instance
    """
    tracker = ProgressTracker(total_phases=total_phases)
    tracker.start(description=description)

    try:
        yield tracker
        tracker.finish()
    except Exception as e:
        tracker.error(f"Error: {e}")
        raise


def display_phase_summary(phase_name: str, stats: dict):
    """
    Display a summary of a completed phase.

    Args:
        phase_name: Name of the completed phase
        stats: Dictionary of statistics to display
    """
    if sys.stdout.isatty():
        console.print(f"\n[bold cyan]{phase_name} Summary:[/bold cyan]")
        for key, value in stats.items():
            console.print(f"  • {key}: [yellow]{value}[/yellow]")
    else:
        logger.info(f"{phase_name} Summary: {stats}")


def display_warning(message: str):
    """
    Display a warning message.

    Args:
        message: Warning message
    """
    if sys.stdout.isatty():
        console.print(f"[bold yellow]⚠[/bold yellow] {message}")
    else:
        logger.warning(message)


def display_error(message: str):
    """
    Display an error message.

    Args:
        message: Error message
    """
    if sys.stdout.isatty():
        console.print(f"[bold red]✗[/bold red] {message}")
    else:
        logger.error(message)


def display_success(message: str):
    """
    Display a success message.

    Args:
        message: Success message
    """
    if sys.stdout.isatty():
        console.print(f"[bold green]✓[/bold green] {message}")
    else:
        logger.info(message)
