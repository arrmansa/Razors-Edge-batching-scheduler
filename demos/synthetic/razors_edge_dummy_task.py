"""Dummy Razors Edge task that keeps the full batching pipeline for tests."""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import Any

import numpy as np

from razors_edge.razors_edge_compute_task import RazorsEdgeComputeTask


class RazorsEdgeDummyTask(RazorsEdgeComputeTask):
    """Dummy task with realistic benchmarking, batching, and post-processing behavior."""

    @property
    def batch_benchmark_sizes(self) -> list[int]:
        return [1, 2, 3, 5, 8]

    @property
    def min_input_size(self) -> int:
        return 1

    @property
    def max_input_size(self) -> int:
        return 1000

    @property
    def max_input_points(self) -> int:
        return 7

    @property
    def is_gpu(self) -> bool:
        return False

    def get_input_size(self, input_data: Any, preprocessed_input: Any) -> int:
        """Return the token count for pre-tokenized model input."""
        return len(input_data)#int(preprocessed_input[1]["input_ids"].shape[1])

    def generate_test_input(self, batch_size: int, input_size: int) -> tuple[tuple, dict[str, np.ndarray]]:
        encoding = self.tokenizer.encode(["A" * input_size] * batch_size)
        return (), encoding

    def load_model(self, model_pool: ThreadPoolExecutor) -> Any:
        from demos.synthetic.dummy_model_and_tokenizer import DummyTokenizer, dummy_model_encode

        self.tokenizer = DummyTokenizer
        max_batch_size = self.batch_benchmark_sizes[-1]
        max_input_size = self.max_input_size
        self.token_buffer = cycle(
            [self.generate_test_input(max_batch_size, max_input_size)[1] for _ in range(model_pool._max_workers + 1)]
        )
        return dummy_model_encode

    def preprocess_input_without_size(self, input_data: str) -> tuple[str, dict[str, np.ndarray]]:
        return input_data, self.tokenizer.encode([input_data])

    def create_batch(self, to_batch: list[tuple[str, dict[str, np.ndarray]]]) -> tuple[tuple, dict[str, np.ndarray]]:
        token_buffer = next(self.token_buffer)
        max_size = max(payload[1]["input_ids"].shape[1] for payload in to_batch)
        batch_size = len(to_batch)
        buffer_copy = {k: v[:batch_size, :max_size].copy() for k, v in token_buffer.items()}

        buffer_copy["input_ids"].fill(0)
        buffer_copy["attention_mask"].fill(0)

        for row, payload in enumerate(to_batch):
            for key, value in payload[1].items():
                buffer_copy[key][row, : value.shape[1]] = value[0]

        return (), buffer_copy

    def postprocess_output(self, call_output: Any) -> Iterable[list[float]]:
        """Normalize embeddings and return list rows."""
        array = np.asarray(call_output, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (array / norms).tolist()
