import unittest
from concurrent.futures import ThreadPoolExecutor
from batching_executor.base_batched_compute_task import BaseBatchedComputeTask


class SampleTask(BaseBatchedComputeTask):
    def __init__(self, model_pool: ThreadPoolExecutor) -> None:
        pass

    def __call__(self, batched_inputs):
        return batched_inputs

class TestBaseBatchedComputeTask(unittest.TestCase):

    def test_preprocess_input_default_passthrough(self):
        task = SampleTask(ThreadPoolExecutor(1))
        self.assertEqual(task.preprocess_input({"a": 1}), {"a": 1})

    def test_accept_in_batch_default_false(self):
        task = SampleTask(ThreadPoolExecutor(1))
        self.assertFalse(task._accept_in_batch([], "candidate"))

    def test_get_batch_ids_list_and_batch_default_behavior(self):
        task = SampleTask(ThreadPoolExecutor(1))
        queue = {(1, 1): "a", (2, 2): "bb"}
        ids, batch = task.get_batch_ids_list_and_batch(queue)
        self.assertEqual(ids, [(1, 1)])
        self.assertEqual(batch, ["a"])

    def test_postprocess_output_default_passthrough(self):
        task = SampleTask(ThreadPoolExecutor(1))
        output = [1, 2, 3]
        self.assertEqual(list(task.postprocess_output(output)), output)


if __name__ == "__main__":
    unittest.main()
