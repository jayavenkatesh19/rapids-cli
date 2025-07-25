# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rich-based terminal GPU monitoring dashboard."""

import time
from typing import List, Optional
from collections import deque

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich import box

from .metrics import MetricsCollector, GPUMetrics, SystemMetrics, NVLinkMetrics


def format_bytes(bytes_value: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.1f} TB"


def format_bytes_per_second(bytes_per_sec: int) -> str:
    """Format bytes per second as human-readable string."""
    return f"{format_bytes(bytes_per_sec)}/s"


def create_progress_bar(value: float, max_value: float = 100.0, width: int = 20) -> str:
    """Create a text-based progress bar."""
    filled = int((value / max_value) * width)
    bar = "█" * filled + "░" * (width - filled)
    return bar


def create_adaptive_sparkline(data: List[float], min_range: float = 5.0, min_value: float = 0.0, max_value: float = 100.0) -> str:
    """Create a sparkline that auto-scales to show variations in the data range.
    
    Args:
        data: List of values to visualize
        min_range: Minimum range to display (expands small variations)
        min_value: Minimum possible value (e.g., 0 for percentages)
        max_value: Maximum possible value (e.g., 100 for percentages)
    
    Returns:
        Sparkline with range information for context
    """
    if not data or len(data) < 2:
        return ""
    
    # Get actual data range
    data_min = min(data)
    data_max = max(data)
    data_range = data_max - data_min
    
    # If range is too small, expand it to show meaningful variations
    if data_range < min_range:
        center = (data_min + data_max) / 2
        data_min = center - min_range / 2
        data_max = center + min_range / 2
        
        # Apply bounds to ensure realistic ranges
        data_min = max(data_min, min_value)  # Don't go below minimum (e.g., 0%)
        data_max = min(data_max, max_value)  # Don't go above maximum (e.g., 100%)
        
        # If we hit a bound, adjust the other end to maintain min_range if possible
        if data_min == min_value:
            data_max = min(data_min + min_range, max_value)
        elif data_max == max_value:
            data_min = max(data_max - min_range, min_value)
        
        data_range = data_max - data_min
    
    # Scale to sparkline characters
    chars = "▁▂▃▄▅▆▇█"
    normalized = []
    for val in data:
        if data_range > 0:
            # Scale to 0-7 range based on actual data range
            normalized_val = int(((val - data_min) / data_range) * 7)
            normalized_val = max(0, min(7, normalized_val))
        else:
            normalized_val = 3  # Middle character for flat data
        normalized.append(normalized_val)
    
    sparkline = "".join(chars[val] for val in normalized)
    
    # Add range info for context
    range_info = f"({data_min:.1f}-{data_max:.1f}%)"
    return f"{sparkline} {range_info}"


def create_sparkline(data: List[float], max_val: float = 100.0) -> str:
    """Create a traditional sparkline chart from data (kept for backward compatibility)."""
    if not data:
        return ""
    
    chars = "▁▂▃▄▅▆▇█"
    normalized = []
    for val in data:
        normalized_val = min(7, int((val / max_val) * 7))
        normalized.append(normalized_val)
    
    return "".join(chars[val] for val in normalized)


class RichGPUDashboard:
    """Rich-based GPU monitoring dashboard."""
    
    def __init__(self, update_interval: float = 1.0):
        self.console = Console()
        self.collector = MetricsCollector()
        self.update_interval = update_interval
        self.start_time = time.time()
        self.update_count = 0
        
        # History for sparklines
        self.gpu_util_history = [deque(maxlen=30) for _ in range(self.collector.ngpus)]
        self.gpu_mem_history = [deque(maxlen=30) for _ in range(self.collector.ngpus)]
        self.cpu_history = deque(maxlen=30)
        self.system_mem_history = deque(maxlen=30)
    
    def create_header_panel(self) -> Panel:
        """Create the header panel."""
        runtime = time.time() - self.start_time
        runtime_str = f"{int(runtime // 60):02d}:{int(runtime % 60):02d}"
        
        title = "[bold blue]🚀 RAPIDS GPU Profiler[/bold blue]"
        status = f"[green]●[/green] Live • Runtime: [cyan]{runtime_str}[/cyan] • Updates: [yellow]{self.update_count}[/yellow]"
        
        header_text = f"{title}\n{status}"
        return Panel(header_text, box=box.DOUBLE)
    
    def create_gpu_table(self, gpu_metrics: List[GPUMetrics]) -> Panel:
        """Create GPU metrics table."""
        if not gpu_metrics:
            return Panel(
                "[red]No GPUs detected![/red]\n\nPlease ensure NVIDIA drivers are installed",
                title="GPU Error",
                box=box.ROUNDED
            )
        
        # Calculate the time span represented by the sparkline
        history_points = self.gpu_util_history[0].maxlen if self.gpu_util_history else 30
        time_span = history_points * self.update_interval
        history_header = f"History ({int(time_span)}s)"
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("GPU", style="cyan", width=4)
        table.add_column("Utilization", width=25)
        table.add_column("Memory", width=25)
        table.add_column("Temp", style="yellow", width=6)
        table.add_column("Power", style="red", width=8)
        table.add_column(history_header, width=35)  # Wider for range info
        
        for i, gpu in enumerate(gpu_metrics):
            # Update history
            self.gpu_util_history[i].append(gpu.utilization)
            memory_percent = (gpu.memory_used / gpu.memory_total) * 100
            self.gpu_mem_history[i].append(memory_percent)
            
            # Create progress bars
            util_bar = create_progress_bar(gpu.utilization, 100, 15)
            mem_bar = create_progress_bar(memory_percent, 100, 15)
            
            # Adaptive sparkline with proper bounds for utilization (0-100%)
            util_spark = create_adaptive_sparkline(
                list(self.gpu_util_history[i]), 
                min_range=5.0, 
                min_value=0.0, 
                max_value=100.0
            )
            
            table.add_row(
                str(gpu.gpu_id),
                f"{util_bar} {gpu.utilization:5.1f}%",
                f"{mem_bar} {format_bytes(gpu.memory_used)}/{format_bytes(gpu.memory_total)}",
                f"{gpu.temperature}°C" if gpu.temperature > 0 else "N/A",
                f"{gpu.power_draw:.1f}W" if gpu.power_draw > 0 else "N/A",
                util_spark
            )
        
        return Panel(table, title="GPU Status", box=box.ROUNDED)
    
    def create_throughput_panel(self, gpu_metrics: List[GPUMetrics], nvlink_metrics: Optional[NVLinkMetrics]) -> Panel:
        """Create throughput panel."""
        total_pcie_rx = sum(gpu.pcie_rx for gpu in gpu_metrics)
        total_pcie_tx = sum(gpu.pcie_tx for gpu in gpu_metrics)
        
        content = f"[cyan]PCIe Throughput[/cyan]\n"
        content += f"  [green]↓ RX:[/green] {format_bytes_per_second(total_pcie_rx)}\n"
        content += f"  [red]↑ TX:[/red] {format_bytes_per_second(total_pcie_tx)}"
        
        if nvlink_metrics and (nvlink_metrics.rx_total > 0 or nvlink_metrics.tx_total > 0):
            content += f"\n\n[magenta]NVLink Throughput[/magenta]\n"
            content += f"  [green]↓ RX:[/green] {format_bytes_per_second(nvlink_metrics.rx_total)}\n"
            content += f"  [red]↑ TX:[/red] {format_bytes_per_second(nvlink_metrics.tx_total)}"
        elif self.collector.nvlink_available:
            content += f"\n\n[magenta]NVLink Available[/magenta]\n"
            content += f"  [dim]No traffic detected[/dim]"
        
        return Panel(content, title="Interconnect", box=box.ROUNDED)
    
    def create_system_panel(self, system_metrics: SystemMetrics) -> Panel:
        """Create system resources panel."""
        self.cpu_history.append(system_metrics.cpu_percent)
        memory_percent = (system_metrics.memory_used / system_metrics.memory_total) * 100
        self.system_mem_history.append(memory_percent)
        
        cpu_bar = create_progress_bar(system_metrics.cpu_percent, 100, 15)
        mem_bar = create_progress_bar(memory_percent, 100, 15)
        
        # CPU utilization with proper bounds (0-100%)
        cpu_spark = create_adaptive_sparkline(
            list(self.cpu_history),
            min_range=5.0,
            min_value=0.0, 
            max_value=100.0
        )
        
        content = f"[yellow]CPU:[/yellow] {cpu_bar} {system_metrics.cpu_percent:5.1f}%\n"
        content += f"[blue]RAM:[/blue] {mem_bar} {format_bytes(system_metrics.memory_used)}/{format_bytes(system_metrics.memory_total)}\n"
        content += f"[dim]CPU History:[/dim] {cpu_spark}\n\n"
        content += f"[red]Disk:[/red] R: {format_bytes_per_second(system_metrics.disk_read_rate)} "
        content += f"W: {format_bytes_per_second(system_metrics.disk_write_rate)}\n"
        content += f"[cyan]Net:[/cyan] R: {format_bytes_per_second(system_metrics.network_rx_rate)} "
        content += f"W: {format_bytes_per_second(system_metrics.network_tx_rate)}"
        
        return Panel(content, title="System Resources", box=box.ROUNDED)
    
    def create_layout(self) -> Layout:
        """Create the complete dashboard layout."""
        gpu_metrics = self.collector.get_gpu_metrics()
        system_metrics = self.collector.get_system_metrics()
        nvlink_metrics = self.collector.get_nvlink_metrics()
        
        self.update_count += 1
        
        layout = Layout()
        layout.split_column(
            Layout(self.create_header_panel(), name="header", size=4),
            Layout(name="main"),
            Layout(Panel("[dim]Press [bold]Ctrl+C[/bold] to exit[/dim]", box=box.ROUNDED), name="footer", size=3)
        )
        
        layout["main"].split_column(
            Layout(self.create_gpu_table(gpu_metrics), name="gpu", ratio=2),
            Layout(name="bottom")
        )
        
        layout["bottom"].split_row(
            Layout(self.create_throughput_panel(gpu_metrics, nvlink_metrics), name="throughput"),
            Layout(self.create_system_panel(system_metrics), name="system")
        )
        
        return layout


def run_dashboard(update_interval: float = 1.0):
    """Run the Rich-based GPU monitoring dashboard."""
    dashboard = RichGPUDashboard(update_interval)
    
    try:
        with Live(dashboard.create_layout(), refresh_per_second=1/update_interval, screen=True) as live:
            while True:
                time.sleep(update_interval)
                live.update(dashboard.create_layout())
    except KeyboardInterrupt:
        dashboard.console.print("\n[yellow]GPU monitoring stopped by user[/yellow]")