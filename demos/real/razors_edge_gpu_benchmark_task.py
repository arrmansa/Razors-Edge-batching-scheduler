"""Run bge-m3 for benchmarking with razors edge batching"""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import Any

import numpy as np

from razors_edge.razors_edge_compute_task import RazorsEdgeComputeTask


class RazorsEdgeGPUBenchmarkTask(RazorsEdgeComputeTask):
    """Dummy task with realistic benchmarking, batching, and post-processing behavior."""

    @property
    def batch_benchmark_sizes(self) -> list[int]:
        return [1, 2, 3, 5, 8, 10, 13, 16]

    @property
    def min_input_size(self) -> int:
        return 1

    @property
    def max_input_size(self) -> int:
        return 1024

    @property
    def max_input_points(self) -> int:
        return 7

    @property
    def is_gpu(self) -> bool:
        return True

    def get_input_size(self, input_data: Any, preprocessed_input: Any) -> int:
        """Return the token count for pre-tokenized model input."""
        return int(preprocessed_input["input_ids"].shape[1])

    def generate_test_input(self, batch_size: int, input_size: int) -> tuple[tuple, dict[str, np.ndarray]]:
        return (), {
            "input_ids": self.torch.ones((batch_size, input_size), dtype=self.torch.long, device="cuda"),
            "attention_mask": self.torch.ones((batch_size, input_size), dtype=self.torch.long, device="cuda")
        }

    def load_model(self, model_pool: ThreadPoolExecutor) -> Any:
        import os
        BASE_DIR = "E:\\Github\\Razors-Edge-batching-scheduler"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HOME"] = f"{BASE_DIR}\\models"
        import torch
        assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"
        from transformers import AutoTokenizer, AutoModel
        with torch.inference_mode():
            self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
            model = AutoModel.from_pretrained("BAAI/bge-m3")
            model = model.eval().half().to("cuda")
        self.torch = torch
        max_batch_size = self.batch_benchmark_sizes[-1]
        max_input_size = self.max_input_size
        self.token_buffer = cycle(
            [self.generate_test_input(max_batch_size, max_input_size)[1] for _ in range(model_pool._max_workers + 1)]
        )
        torch.cuda.empty_cache()
        def run_model(*_, **inputs):
            with torch.inference_mode(), torch.autocast("cuda"):
                return model(**inputs)
        return run_model

    def preprocess_input_without_size(self, input_data: str) -> tuple[str, dict[str, np.ndarray]]:
        return self.tokenizer([input_data], padding=True, truncation=True, return_tensors="pt")

    def create_batch(self, to_batch: list[tuple[str, dict[str, np.ndarray]]]) -> tuple[tuple, dict[str, np.ndarray]]:
        token_buffer = next(self.token_buffer)
        max_size = max(payload["input_ids"].shape[1] for payload in to_batch)
        batch_size = len(to_batch)
        buffer_copy = {k: v[:batch_size, :max_size] for k, v in token_buffer.items()}
        buffer_copy["input_ids"].fill_(1)
        buffer_copy["attention_mask"].fill_(0)
        for row, payload in enumerate(to_batch):
            for key, value in payload.items():
                buffer_copy[key][row, : value.shape[1]] = value[0]
        return (), buffer_copy

    def postprocess_output(self, call_output: Any) -> Iterable[list[float]]:
        """Normalize embeddings and return list rows."""
        with self.torch.inference_mode() and self.torch.autocast("cuda"):
            embeddings = call_output.last_hidden_state.mean(dim=1)
            embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.tolist()
