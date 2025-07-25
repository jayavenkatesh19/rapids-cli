# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Rapids CLI is a command-line interface for RAPIDS."""

import rich_click as click
from rich.console import Console
from rich.traceback import install

from rapids_cli.doctor import doctor_check
from rapids_cli.profile import run_dashboard

console = Console()
install(show_locals=True)


@click.group()
def rapids():
    """The Rapids CLI is a command-line interface for RAPIDS."""
    pass


@rapids.command()
@click.option(
    "--verbose", is_flag=True, help="Enable verbose mode for detailed output."
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Perform a dry run without making any changes.",
)
@click.argument("filters", nargs=-1)
def doctor(verbose, dry_run, filters):
    """Run health checks to ensure RAPIDS is installed correctly."""
    status = doctor_check(verbose, dry_run, filters)
    if not status:
        raise click.ClickException("Health checks failed.")

@rapids.command()
@click.option(
    "--interval",
    "-i", 
    default=1.0,
    help="Update interval in seconds (default: 1.0)"
)
def profile(interval):
    """Monitor GPU usage and system resources in real-time.
    
    Launch a terminal-based dashboard showing GPU utilization, memory usage,
    throughput metrics, and system resources. Perfect for monitoring your
    RAPIDS workloads!
    
    Examples:
        rapids profile                    # Start dashboard with 1s updates
        rapids profile --interval 0.5     # Fast updates (500ms)
        rapids profile -i 2.0             # Slower updates (2s)
    """
    try:
        console.print(f"[green]Starting RAPIDS GPU profiler (interval: {interval}s)[/green]")
        console.print("[dim]Press Ctrl+C to exit[/dim]")
        run_dashboard(update_interval=interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]GPU monitoring stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    rapids()
