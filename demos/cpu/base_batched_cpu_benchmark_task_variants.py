"""CPU base batched task variants with fixed batch caps."""

from demos.cpu.base_batched_cpu_benchmark_task import BaseBatchedCPUBenchmarkTask


class BaseBatchedCPUBenchmarkTaskB2(BaseBatchedCPUBenchmarkTask):
    max_batch_size = 2


class BaseBatchedCPUBenchmarkTaskB3(BaseBatchedCPUBenchmarkTask):
    max_batch_size = 3


class BaseBatchedCPUBenchmarkTaskB4(BaseBatchedCPUBenchmarkTask):
    max_batch_size = 4
