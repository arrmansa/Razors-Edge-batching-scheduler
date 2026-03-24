"""Synthetic dummy task variants for latency strategy comparison."""

from typing import Literal

from demos.synthetic.razors_edge_dummy_task import RazorsEdgeDummyTask


class RazorsEdgeDummyTaskRMS(RazorsEdgeDummyTask):
    """Dummy task using the default RMS latency strategy."""

    @property
    def latency_strategy(self) -> Literal["RMS", "FIFO", "MINMAX"]:
        return "RMS"


class RazorsEdgeDummyTaskFIFO(RazorsEdgeDummyTask):
    """Dummy task using FIFO (oldest-first) latency strategy."""

    @property
    def latency_strategy(self) -> Literal["RMS", "FIFO", "MINMAX"]:
        return "FIFO"

class RazorsEdgeDummyTaskMinMax(RazorsEdgeDummyTask):
    """Dummy task using MINMAX latency strategy."""

    @property
    def latency_strategy(self) -> Literal["RMS", "FIFO", "MINMAX"]:
        return "MINMAX"
