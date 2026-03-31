import unittest

import numpy as np

from razors_edge.optimal_benchmarking import (
    MINIMUM_SPLINE_POINTS,
    create_batch_timing_estimators,
    generate_benchmark_points,
    get_benchmark_data_paddings,
    get_points_ratio,
)


class TestOptimalBenchmarking(unittest.TestCase):
    def test_get_points_ratio_returns_positive_float(self):
        ratio = get_points_ratio(min_tokens=1, max_tokens=101, max_points=11)
        self.assertGreater(ratio, 0)
        self.assertIsInstance(ratio, float)

    def test_generate_benchmark_points_respects_bounds(self):
        points = generate_benchmark_points(min_tokens=1, max_tokens=100, points_ratio=0.4)
        self.assertGreaterEqual(len(points), MINIMUM_SPLINE_POINTS)
        self.assertEqual(points[0], 1)
        self.assertEqual(points[-1], 100)
        self.assertTrue(all(a <= b for a, b in zip(points, points[1:])))

    def test_get_benchmark_data_paddings_for_equal_last_point(self):
        pad_points, pad_timings = get_benchmark_data_paddings(
            prev_benchmark_points=[1, 4, 9, 16],
            prev_benchmark_timings=[10, 20, 30, 40],
            new_benchmark_points=[1, 4, 9, 16],
            new_benchmark_timings=[9, 18, 27, 36],
            ratio=5/4,
        )
        self.assertEqual(pad_points, [])
        self.assertEqual(pad_timings, [])


    def test_get_benchmark_data_paddings_raises_for_invalid_ranges(self):
        with self.assertRaises(RuntimeError):
            get_benchmark_data_paddings(
                prev_benchmark_points=[1, 2, 3],
                prev_benchmark_timings=[10, 20, 30],
                new_benchmark_points=[1, 2, 4],
                new_benchmark_timings=[9, 18, 27],
                ratio=1.0,
            )

    def test_create_batch_timing_estimators_interpolates_missing_batch_size(self):
        batch_benchmark_points = [1, 3]
        batch_timing_data = [
            ([1, 4, 9, 16], [10, 20, 30, 40]),
            ([1, 4, 9, 16], [30, 40, 50, 60]),
        ]

        estimators = create_batch_timing_estimators(
            batch_benchmark_points=batch_benchmark_points,
            batch_timing_data=batch_timing_data,
            max_tokens=16,
        )

        self.assertEqual(estimators.shape, (3, 17))
        # batch size 2 should be linearly interpolated between batch sizes 1 and 3
        self.assertTrue(np.all(estimators[1] >= estimators[0]))
        self.assertTrue(np.all(estimators[1] <= estimators[2]))
    def test_create_batch_timing_estimators_shape(self):
        batch_benchmark_points = [1, 3]
        batch_timing_data = [
            ([1, 4, 9, 16], [10, 15, 20, 25]),
            ([1, 4, 9, 16], [25, 35, 45, 55]),
        ]

        estimators = create_batch_timing_estimators(
            batch_benchmark_points=batch_benchmark_points,
            batch_timing_data=batch_timing_data,
            max_tokens=16,
        )

        self.assertEqual(estimators.shape[0], 3)
        self.assertEqual(estimators.shape[1], 17)
        self.assertTrue(np.all(estimators > 0))


if __name__ == "__main__":
    unittest.main()
