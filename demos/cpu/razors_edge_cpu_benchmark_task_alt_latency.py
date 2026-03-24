from typing import Literal

from demos.cpu.razors_edge_cpu_benchmark_task import RazorsEdgeCPUBenchmarkTask


class RazorsEdgeCPUBenchmarkRMSTask(RazorsEdgeCPUBenchmarkTask):
    """CPU benchmark task using the default RMS latency strategy."""

    @property
    def latency_strategy(self) -> Literal["RMS", "FIFO", "MINMAX"]:
        return "RMS"


class RazorsEdgeCPUBenchmarkFIFOTask(RazorsEdgeCPUBenchmarkTask):
    """CPU benchmark task using FIFO (oldest-first) latency strategy."""

    @property
    def latency_strategy(self) -> Literal["RMS", "FIFO", "MINMAX"]:
        return "FIFO"


class RazorsEdgeCPUBenchmarkMinMaxTask(RazorsEdgeCPUBenchmarkTask):
    """CPU benchmark task using MINMAX latency strategy."""

    @property
    def latency_strategy(self) -> Literal["RMS", "FIFO", "MINMAX"]:
        return "MINMAX"
