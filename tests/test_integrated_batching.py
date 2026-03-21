"""Integrated batch-size tests covering both sync and async executor paths."""

import asyncio
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from src.executor.base_batched_compute_task import BaseBatchedComputeTask
from src.executor.process_manager import ComputeExecutor


class BatchingDetectionTask(BaseBatchedComputeTask):
    """Task that returns the batch size for every item in that batch."""

    def __init__(self, model_pool):
        self.model_pool = model_pool

    def __call__(self, batched_inputs):
        # Create a small processing window so requests queue up together.
        time.sleep(0.1)
        return [len(batched_inputs)] * len(batched_inputs)

    def _accept_in_batch(self, current_batch, candidate):
        return len(current_batch) < 4


class TestBaseBatchedComputeTask(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.executor = ComputeExecutor([BatchingDetectionTask], async_limit=16, model_thread_limit=1)

    async def test_async_compute_batches_requests(self):
        results = await asyncio.gather(
            *(self.executor.async_compute_fn(BatchingDetectionTask, i) for i in range(8))
        )

        self.assertTrue(all(1 <= result <= 4 for result in results))
        self.assertIn(4, results)

    def test_sync_compute_batches_requests(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(self.executor.sync_compute_fn, BatchingDetectionTask, i) for i in range(8)]
            results = [future.result() for future in futures]

        self.assertTrue(all(1 <= result <= 4 for result in results))
        self.assertIn(4, results)

    def tearDown(self):
        del self.executor
        return None


if __name__ == "__main__":
    unittest.main()
