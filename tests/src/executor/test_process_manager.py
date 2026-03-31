import unittest
from batching_executor.process_manager import ComputeExecutor


class TestComputeExecutorInternals(unittest.TestCase):
    def test_choose_fair_queue_uses_smallest_operation_id(self):
        queues = [{(2, 100): "a"}, {(1, 200): "b"}, {}]
        chosen = ComputeExecutor._InternalProcess.choose_fair_queue(queues)
        self.assertEqual(chosen, 1)

    def test_remove_ids_from_queue_removes_only_requested(self):
        queue = {(1, 11): "a", (2, 22): "b", (3, 33): "c"}
        ComputeExecutor._InternalProcess.remove_ids_from_queue(queue, [(1, 11), (3, 33)])
        self.assertEqual(queue, {(2, 22): "b"})


if __name__ == "__main__":
    unittest.main()
