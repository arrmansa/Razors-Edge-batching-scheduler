"""Dummy Batched task that keeps the full batching pipeline for tests."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any
from batching_executor.base_batched_compute_task import BaseBatchedComputeTask
import numpy as np


class BaseBatchedDummyTask(BaseBatchedComputeTask):
    """Dummy task with realistic simple batching, and post-processing behavior."""

    max_batch_size = 1

    def __init__(self, model_pool: ThreadPoolExecutor) -> Any:
        from demos.synthetic.dummy_model_and_tokenizer import DummyTokenizer, dummy_model_encode
        self.tokenizer = DummyTokenizer
        self.model = dummy_model_encode

    def _accept_in_batch(self, current_batch: list, candidate: Any) -> bool:
        """Check if a candidate belongs to the current batch based on text size, current size etc. False by default for batch size 1."""
        return len(current_batch) < self.max_batch_size

    def __call__(self, batched_inputs):
        return self.model(**self.tokenizer.encode(batched_inputs))

    def postprocess_output(self, call_output: Any) -> list[list[float]]:
        """Normalize embeddings and return list rows."""
        array = np.asarray(call_output, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (array / norms).tolist()
