"""Run jina-embeddings-v2-base-en for benchmarking with razors edge batching"""

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from typing import Any
import logging

import numpy as np
from razors_edge.razors_edge_compute_task import RazorsEdgeComputeTask


class RazorsEdgeCPUBenchmarkTask(RazorsEdgeComputeTask):
    """Jina-v2 benchmarking task optimized for CPU execution."""

    
    thread_benchmark_points = [8]#[1, 2, 4, 8]
    thread_acceptable_inefficiency = 1.05

    @property
    def batch_benchmark_sizes(self) -> list[int]:
        # CPU batching usually hits diminishing returns faster than GPU
        return [1, 2, 3, 4, 6]

    @property
    def min_input_size(self) -> int:
        return 1

    @property
    def max_input_size(self) -> int:
        # Jina-v2 supports 8192, but for benchmarking we can stick to 512 or higher
        return 512

    @property
    def max_input_points(self) -> int:
        return 5

    @property
    def is_gpu(self) -> bool:
        # Changed to False for CPU execution
        return False

    def get_input_size(self, input_data: Any, preprocessed_input: Any) -> int:
        return int(preprocessed_input["input_ids"].shape[1])

    def generate_test_input(self, batch_size: int, input_size: int) -> tuple[tuple, dict[str, np.ndarray]]:
        # Using CPU tensors instead of CUDA
        return (), {
            "input_ids": self.torch.ones((batch_size, input_size), dtype=self.torch.long, device="cpu"),
            "attention_mask": self.torch.ones((batch_size, input_size), dtype=self.torch.long, device="cpu"),
            "token_type_ids": self.torch.ones((batch_size, input_size), dtype=self.torch.long, device="cpu"),
        }

    def load_model(self, model_pool: ThreadPoolExecutor) -> Any:
        import os
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        # Environment setup
        BASE_DIR = "E:\\Github\\Razors-Edge-batching-scheduler"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HOME"] = f"{BASE_DIR}\\models"
        
        self.torch = torch
        
        # Jina-v2 requires trust_remote_code=True for their custom architecture
        with torch.inference_mode():
            self.tokenizer = AutoTokenizer.from_pretrained(
                "jinaai/jina-embeddings-v2-base-en", 
                trust_remote_code=True
            )
            model = AutoModel.from_pretrained(
                "jinaai/jina-embeddings-v2-base-en", 
                trust_remote_code=True
            )
            # Switch to eval mode and keep on CPU (float32 is standard for CPU)
            model = model.eval()
            
        max_batch_size = self.batch_benchmark_sizes[-1]
        max_input_size = self.max_input_size
        
        # Pre-fill token buffer on CPU
        self.token_buffer = cycle(
            [self.generate_test_input(max_batch_size, max_input_size)[1] for _ in range(model_pool._max_workers + 1)]
        )

        def run_model(*_, **inputs):
            with torch.inference_mode():
                # Removed autocast("cuda") as it's not applicable for standard CPU runs
                return model(**inputs)
        
        # Set optimal threads
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        thread_benchmark_input = self.generate_test_input(4, 256)
        thread_timings: list[tuple[float, int]] = []
        with torch.inference_mode():
            run_model(*thread_benchmark_input[0], **thread_benchmark_input[1])
            for threadcount in self.thread_benchmark_points:
                logging.warning(f"Setting thread to {threadcount}")
                self.torch.set_num_threads(threadcount)
                thread_timings.append((self.model_test_pattern(lambda: run_model(*thread_benchmark_input[0], **thread_benchmark_input[1])), threadcount))

        logging.warning(f"Thread timings: {thread_timings}")
        thread_count = sorted(filter(lambda x: x[0] < self.thread_acceptable_inefficiency * min(thread_timings)[0], thread_timings), key=lambda x: x[1])[0][1]
        self.torch.set_num_threads(thread_count)
        logging.warning(f"Thread count set to: {thread_count}")

        return run_model

    def preprocess_input_without_size(self, input_data: str) -> tuple[str, dict[str, np.ndarray]]:
        # Ensure return_tensors is "pt" (PyTorch) and sits on CPU
        return self.tokenizer([input_data], padding=True, truncation=True, return_tensors="pt")

    def create_batch(self, to_batch: list[tuple[str, dict[str, np.ndarray]]]) -> tuple[tuple, dict[str, np.ndarray]]:
        token_buffer = next(self.token_buffer)
        max_size = max(payload["input_ids"].shape[1] for payload in to_batch)
        batch_size = len(to_batch)
        
        buffer_copy = {k: v[:batch_size, :max_size] for k, v in token_buffer.items()}
        buffer_copy["input_ids"].fill_(0)
        buffer_copy["attention_mask"].fill_(0)
        buffer_copy["token_type_ids"].fill_(0)
        
        for row, payload in enumerate(to_batch):
            for key, value in payload.items():
                buffer_copy[key][row, : value.shape[1]] = value[0]
        return (), buffer_copy

    def postprocess_output(self, call_output: Any) -> Iterable[list[float]]:
        with self.torch.inference_mode():
            # Jina-v2 usually uses mean pooling on the last hidden state
            embeddings = call_output.last_hidden_state.mean(dim=1)
            embeddings = self.torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.tolist()
