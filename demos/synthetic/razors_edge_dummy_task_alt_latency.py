"""Synthetic dummy task variants for latency strategy comparison."""

from typing import Literal

from demos.synthetic.razors_edge_dummy_task import RazorsEdgeDummyTask


class RazorsEdgeDummyTaskDefault(RazorsEdgeDummyTask):
    """Dummy task using the default MINMAX latency strategy."""


class RazorsEdgeDummyTaskFIFO(RazorsEdgeDummyTask):
    """Dummy task using FIFO (oldest-first) latency strategy."""

    @property
    def latency_strategy(self) -> Literal["FIFO", "MINMAX", "GUARDED_BATCH_SIZE"]:
        return "FIFO"


class RazorsEdgeDummyTaskBatchSize(RazorsEdgeDummyTask):
    """Dummy task using GUARDED_BATCH_SIZE latency strategy."""

    @property
    def latency_strategy(self) -> Literal["FIFO", "MINMAX", "GUARDED_BATCH_SIZE"]:
        return "GUARDED_BATCH_SIZE"
