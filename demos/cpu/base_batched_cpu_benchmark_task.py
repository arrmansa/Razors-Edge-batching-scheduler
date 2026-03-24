"""Run jina-embeddings-v2-base-en for benchmarking with simple fixed batching."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.executor.base_batched_compute_task import BaseBatchedComputeTask


class BaseBatchedCPUBenchmarkTask(BaseBatchedComputeTask):
    """CPU baseline task with fixed max_batch_size and simple batching policy."""

    max_batch_size = 1

    def __init__(self, model_pool: ThreadPoolExecutor) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        with torch.inference_mode():
            self.tokenizer = AutoTokenizer.from_pretrained(
                "jinaai/jina-embeddings-v2-base-en",
                trust_remote_code=True,
            )
            self.model = AutoModel.from_pretrained(
                "jinaai/jina-embeddings-v2-base-en",
                trust_remote_code=True,
            ).eval()

    def _accept_in_batch(self, current_batch: list, candidate: str) -> bool:
        """Accept requests until max_batch_size is reached."""
        return len(current_batch) < self.max_batch_size

    def __call__(self, batched_inputs: list[str]) -> Any:
        with self.torch.inference_mode():
            encoded = self.tokenizer(batched_inputs, padding=True, truncation=True, return_tensors="pt")
            return self.model(**encoded)

    def postprocess_output(self, call_output: Any) -> list[list[float]]:
        with self.torch.inference_mode():
            embeddings = call_output.last_hidden_state.mean(dim=1)
            embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.tolist()
