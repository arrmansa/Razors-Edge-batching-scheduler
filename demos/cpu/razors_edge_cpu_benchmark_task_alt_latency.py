from typing import Literal

from demos.cpu.razors_edge_cpu_benchmark_task import RazorsEdgeCPUBenchmarkTask


class RazorsEdgeCPUBenchmarkDefaultTask(RazorsEdgeCPUBenchmarkTask):
    """CPU benchmark task using the default MINMAX latency strategy."""


class RazorsEdgeCPUBenchmarkFIFOTask(RazorsEdgeCPUBenchmarkTask):
    """CPU benchmark task using FIFO (oldest-first) latency strategy."""

    @property
    def latency_strategy(self) -> Literal["FIFO", "MINMAX", "GUARDED_BATCH_SIZE"]:
        return "FIFO"


class RazorsEdgeCPUBenchmarkBatchSizeTask(RazorsEdgeCPUBenchmarkTask):
    """CPU benchmark task using GUARDED_BATCH_SIZE latency strategy."""

    @property
    def latency_strategy(self) -> Literal["FIFO", "MINMAX", "GUARDED_BATCH_SIZE"]:
        return "GUARDED_BATCH_SIZE"
