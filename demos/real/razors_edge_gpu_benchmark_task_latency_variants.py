from typing import Literal

from demos.real.razors_edge_gpu_benchmark_task import RazorsEdgeGPUBenchmarkTask


class RazorsEdgeGPUBenchmarkDefaultTask(RazorsEdgeGPUBenchmarkTask):
    """GPU benchmark task using the default MINMAX latency strategy."""


class RazorsEdgeGPUBenchmarkFIFOTask(RazorsEdgeGPUBenchmarkTask):
    """GPU benchmark task using FIFO (oldest-first) latency strategy."""

    @property
    def latency_strategy(self) -> Literal["FIFO", "MINMAX", "GUARDED_BATCH_SIZE"]:
        return "FIFO"


class RazorsEdgeGPUBenchmarkBatchSizeTask(RazorsEdgeGPUBenchmarkTask):
    """GPU benchmark task using GUARDED_BATCH_SIZE latency strategy."""

    @property
    def latency_strategy(self) -> Literal["FIFO", "MINMAX", "GUARDED_BATCH_SIZE"]:
        return "GUARDED_BATCH_SIZE"
