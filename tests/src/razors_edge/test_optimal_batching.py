import unittest

import numpy as np

from src.razors_edge.optimal_batching import _compiled_bit_length, get_batch_start_end_idx_and_duration


class TestOptimalBatching(unittest.TestCase):
    def test_compiled_bit_length_matches_python(self):
        for value in [1, 2, 3, 7, 8, 255, 256, 1024, (1 << 40) - 1]:
            self.assertEqual(_compiled_bit_length(value), value.bit_length())

    def test_get_batch_start_end_idx_and_duration_returns_valid_slice(self):
        sorted_model_input_sizes = (1, 2, 3, 4)
        # batch sizes 1 and 2, token sizes 0..4
        batch_timing_estimators = np.array(
            [
                [3, 3, 3, 3, 3],
                [5, 5, 5, 5, 5],
            ],
            dtype=np.int64,
        )
        queuing_times = (0, 0, 0, 0)

        start_idx, end_idx, duration = get_batch_start_end_idx_and_duration(
            sorted_model_input_sizes=sorted_model_input_sizes,
            batch_timing_estimators=batch_timing_estimators,
            queuing_times=queuing_times,
            expected_schedule_time=0,
        )

        self.assertGreaterEqual(start_idx, 0)
        self.assertLess(end_idx, len(sorted_model_input_sizes) + 1)
        self.assertGreater(end_idx, start_idx)
        self.assertGreater(duration, 0)


if __name__ == "__main__":
    unittest.main()
