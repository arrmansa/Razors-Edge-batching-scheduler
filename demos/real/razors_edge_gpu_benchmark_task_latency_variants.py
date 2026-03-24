from typing import Literal

from demos.real.razors_edge_gpu_benchmark_task import RazorsEdgeGPUBenchmarkTask


class RazorsEdgeGPUBenchmarkRMSTask(RazorsEdgeGPUBenchmarkTask):
    """Dummy task using the default RMS latency strategy."""

    @property
    def latency_strategy(self) -> Literal["RMS", "FIFO", "MINMAX"]:
        return "RMS"

class RazorsEdgeGPUBenchmarkFIFOTask(RazorsEdgeGPUBenchmarkTask):
    """Dummy task using FIFO (oldest-first) latency strategy."""

    @property
    def latency_strategy(self) -> Literal["RMS", "FIFO", "MINMAX"]:
        return "FIFO"

class RazorsEdgeGPUBenchmarkMinMaxTask(RazorsEdgeGPUBenchmarkTask):
    """Dummy task using MINMAX latency strategy."""

    @property
    def latency_strategy(self) -> Literal["RMS", "FIFO", "MINMAX"]:
        return "MINMAX"
