"""Run bge-m3 for benchmarking with simple batching mechanism"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from torch import device

from batching_executor.base_batched_compute_task import BaseBatchedComputeTask


class BaseBatchedGPUBenchmarkTask(BaseBatchedComputeTask):
    """Dummy task with realistic simple batching, and post-processing behavior."""

    max_batch_size = 1

    def __init__(self, model_pool: ThreadPoolExecutor) -> Any:
        import os
        BASE_DIR = "E:\\Github\\Razors-Edge-batching-scheduler"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HOME"] = f"{BASE_DIR}\\models"
        import torch
        assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"
        from transformers import AutoTokenizer, AutoModel
        with torch.inference_mode():
            self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
            self.model = AutoModel.from_pretrained("BAAI/bge-m3")
            self.model = self.model.eval().half().to("cuda")
        self.torch = torch

    def _accept_in_batch(self, current_batch: list, candidate: str) -> bool:
        """Check if a candidate belongs to the current batch based on text size, current size etc. False by default for batch size 1."""
        return len(current_batch) < self.max_batch_size

    def __call__(self, batched_inputs):
        with self.torch.inference_mode(), self.torch.autocast("cuda"):
            return self.model(**{k: v.to("cuda") for k, v in self.tokenizer(batched_inputs, padding=True, truncation=True, return_tensors="pt").items()})

    def postprocess_output(self, call_output: Any) -> list[list[float]]:
        """Normalize embeddings and return list rows."""
        with self.torch.inference_mode(), self.torch.autocast("cuda"):
            embeddings = call_output.last_hidden_state.mean(dim=1)
            embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.tolist()
