# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU and system metrics collection based on nvdashboard patterns."""

import time
from dataclasses import dataclass
from typing import List, Optional
import pynvml
import psutil


@dataclass
class GPUMetrics:
    """Individual GPU metrics."""
    gpu_id: int
    utilization: float          # %
    memory_used: int           # bytes  
    memory_total: int          # bytes
    temperature: int           # Celsius
    power_draw: float          # Watts
    pcie_rx: int              # bytes/sec
    pcie_tx: int              # bytes/sec


@dataclass
class SystemMetrics:
    """System metrics."""
    timestamp: float
    cpu_percent: float
    memory_used: int
    memory_total: int
    disk_read_rate: int         # bytes/sec
    disk_write_rate: int        # bytes/sec
    network_rx_rate: int        # bytes/sec
    network_tx_rate: int        # bytes/sec


@dataclass
class NVLinkMetrics:
    """NVLink throughput metrics."""
    rx_total: int              # bytes/sec
    tx_total: int              # bytes/sec


class MetricsCollector:
    """Collects GPU and system metrics."""
    
    def __init__(self):
        self.ngpus = 0
        self.gpu_handles = []
        self.pci_gen = None
        self.pci_width = None
        self.nvlink_available = False
        
        # State for rate calculations
        self.prev_nvlink_throughput = None
        self.prev_disk_io = None
        self.prev_network_io = None
        self.prev_time = None
        
        self._initialize_gpu()
    
    def _initialize_gpu(self):
        """Initialize GPU monitoring."""
        try:
            pynvml.nvmlInit()
            self.ngpus = pynvml.nvmlDeviceGetCount()
            self.gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) 
                for i in range(self.ngpus)
            ]
            
            if self.ngpus > 0:
                # Get PCIe info
                try:
                    self.pci_gen = pynvml.nvmlDeviceGetMaxPcieLinkGeneration(self.gpu_handles[0])
                    self.pci_width = pynvml.nvmlDeviceGetMaxPcieLinkWidth(self.gpu_handles[0])
                except (pynvml.NVMLError_NotSupported, IndexError):
                    self.pci_gen = None
                    self.pci_width = None
                
                # Check NVLink availability
                try:
                    pynvml.nvmlDeviceGetNvLinkVersion(self.gpu_handles[0], 0)
                    self.nvlink_available = True
                except (IndexError, pynvml.NVMLError_NotSupported):
                    self.nvlink_available = False
                    
        except pynvml.NVMLError_LibraryNotFound:
            pass
    
    def get_gpu_metrics(self) -> List[GPUMetrics]:
        """Get current GPU metrics for all GPUs."""
        if self.ngpus == 0:
            return []
        
        metrics = []
        for i in range(self.ngpus):
            handle = self.gpu_handles[i]
            
            # Basic metrics
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Temperature (if available)
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except pynvml.NVMLError_NotSupported:
                temperature = 0
            
            # Power draw (if available)
            try:
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            except pynvml.NVMLError_NotSupported:
                power_draw = 0.0
            
            # PCIe throughput (if available)
            pcie_rx = pcie_tx = 0
            if self.pci_gen is not None:
                try:
                    pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(
                        handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                    ) * 1024  # KB to bytes
                    pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(
                        handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                    ) * 1024  # KB to bytes
                except pynvml.NVMLError_NotSupported:
                    pass
            
            metrics.append(GPUMetrics(
                gpu_id=i,
                utilization=utilization,
                memory_used=memory_info.used,
                memory_total=memory_info.total,
                temperature=temperature,
                power_draw=power_draw,
                pcie_rx=pcie_rx,
                pcie_tx=pcie_tx
            ))
        
        return metrics
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        now = time.time()
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # Disk I/O rates
        disk_io = psutil.disk_io_counters()
        disk_read_rate = disk_write_rate = 0
        if disk_io and self.prev_disk_io and self.prev_time:
            time_diff = now - self.prev_time
            if time_diff > 0:
                disk_read_rate = int((disk_io.read_bytes - self.prev_disk_io.read_bytes) / time_diff)
                disk_write_rate = int((disk_io.write_bytes - self.prev_disk_io.write_bytes) / time_diff)
        self.prev_disk_io = disk_io
        
        # Network I/O rates
        network_io = psutil.net_io_counters()
        network_rx_rate = network_tx_rate = 0
        if network_io and self.prev_network_io and self.prev_time:
            time_diff = now - self.prev_time
            if time_diff > 0:
                network_rx_rate = int((network_io.bytes_recv - self.prev_network_io.bytes_recv) / time_diff)
                network_tx_rate = int((network_io.bytes_sent - self.prev_network_io.bytes_sent) / time_diff)
        self.prev_network_io = network_io
        
        self.prev_time = now
        
        return SystemMetrics(
            timestamp=now,
            cpu_percent=cpu_percent,
            memory_used=memory.used,
            memory_total=memory.total,
            disk_read_rate=disk_read_rate,
            disk_write_rate=disk_write_rate,
            network_rx_rate=network_rx_rate,
            network_tx_rate=network_tx_rate
        )
    
    def get_nvlink_metrics(self) -> Optional[NVLinkMetrics]:
        """Get NVLink throughput metrics."""
        if not self.nvlink_available or self.ngpus < 2:
            return None
        
        return NVLinkMetrics(
            rx_total=0,
            tx_total=0
        )