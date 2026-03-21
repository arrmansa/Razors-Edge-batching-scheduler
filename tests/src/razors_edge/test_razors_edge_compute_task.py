import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from src.razors_edge.razors_edge_compute_task import RazorsEdgeComputeTask


class DummyRazorsEdgeTask(RazorsEdgeComputeTask):
    @property
    def batch_benchmark_sizes(self) -> list[int]:
        return [1, 2]

    @property
    def min_input_size(self) -> int:
        return 1

    @property
    def max_input_size(self) -> int:
        return 8

    @property
    def max_input_points(self) -> int:
        return 4

    @property
    def is_gpu(self) -> bool:
        return False

    def get_input_size(self, input_data, preprocessed_input) -> int:
        return len(str(preprocessed_input))

    def generate_test_input(self, batch_size: int, input_size: int):
        return (tuple(["x" * input_size] * batch_size), {}), {}

    def load_model(self, model_pool: ThreadPoolExecutor):
        return lambda *args, **kwargs: list(args)

    def create_batch(self, to_batch):
        return (tuple(to_batch), {})

    def get_batch_timing_data(self):
        return [([1, 4], [10, 20]), ([1, 4], [15, 30])]


class TestRazorsEdgeComputeTaskMethods(unittest.TestCase):
    def test_preprocess_input_returns_payload_and_size(self):
        task = DummyRazorsEdgeTask.__new__(DummyRazorsEdgeTask)
        processed, size = task.preprocess_input("hello")
        self.assertEqual(processed, "hello")
        self.assertEqual(size, 5)

    def test_get_batch_ids_list_and_batch_uses_selected_slice(self):
        task = DummyRazorsEdgeTask.__new__(DummyRazorsEdgeTask)
        task.batch_timing_estimators = None
        task.expected_schedule_time = time.perf_counter_ns()
        task.get_batch_start_end_idx_and_duration = lambda sizes, estimators, queueing, expected: (1, 3, 123)
        task.create_batch = lambda items: (tuple(items), {"count": len(items)})

        queue = {
            (100, 10): ("a", 2),
            (101, 11): ("bb", 3),
            (102, 12): ("ccc", 4),
        }
        keys, batch = task.get_batch_ids_list_and_batch(queue)
        self.assertEqual(keys, ((101, 11), (102, 12)))
        self.assertEqual(batch[0], ("bb", "ccc"))
        self.assertEqual(batch[1], {"count": 2})
        self.assertGreaterEqual(task.expected_schedule_time, 123)


if __name__ == "__main__":
    unittest.main()
