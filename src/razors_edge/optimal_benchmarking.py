"""Model benchmarking optimization functions for efficient batch timing estimation and GPU/CPU performance testing."""

import gc
import time

import numpy as np
# import torch torch decoupled from benchmark, should still work
from scipy.interpolate import UnivariateSpline

MINIMUM_SPLINE_POINTS = 4  # Minimum number of points for spline interpolation

def get_points_ratio(min_tokens: int, max_tokens: int, max_points: int) -> float:
    """Calculate the ratio of points to token range for benchmark point generation."""
    return (max_points - 1) / (max_tokens - min_tokens) ** 0.5


def generate_benchmark_points(min_tokens: int, max_tokens: int, points_ratio: float) -> list[int]:
    """Generate a list of benchmark points between min_tokens and max_tokens with square distribution."""
    num_points = max(int((max_tokens - min_tokens) ** 0.5 * points_ratio + 1), MINIMUM_SPLINE_POINTS)
    return np.square(np.linspace(min_tokens**0.5, max_tokens**0.5, num_points)).astype(np.int32).tolist()


def calculate_next_benchmark_points(
    first_benchmark_points: list[int],
    first_benchmark_timings: list[int],
    first_benchmark_scaling: float,
    second_benchmark_points: list[int],
    second_benchmark_timings: list[int],
    second_benchmark_scaling: float,
    points_ratio: float,
    spline_k=3,
    spline_s=1,
    minimum_saturation_ratio=0.95,
    minimum_token_change=0.1,
    ) -> list[int]:
    """
    Calculate the next set of benchmark points based on the timings of two sets of benchmarks.

    The ratio of the second benchmark to the first is generally expected to be a scaled and shifted sigmoid function.
    The shape occurs as initially the memory bandwidth is the bottleneck, then the compute becomes the bottleneck.
    We expect the maximum of the ratio to be around 1 if the run saturates, the minimum is at the start.
    If the ratio does not saturate or the token change is too small, it returns the second benchmark points.
    If the ratio saturates, it generates new benchmark points based on the 2x of the transition point of the sigmoid.
   """
    spline1 = UnivariateSpline(first_benchmark_points, first_benchmark_timings, k=spline_k, s=spline_s)
    spline2 = UnivariateSpline(second_benchmark_points, second_benchmark_timings, k=spline_k, s=spline_s)
    start_x = max(min(first_benchmark_points), min(second_benchmark_points))
    end_x = min(max(first_benchmark_points), max(second_benchmark_points))
    valid_x = np.arange(start_x, end_x + 1, dtype=np.float64)
    valid_y1: np.ndarray = spline1(valid_x)  # type: ignore
    valid_y2: np.ndarray = spline2(valid_x)  # type: ignore
    ratio_sigmoid = (valid_y2 / second_benchmark_scaling) / (valid_y1 / first_benchmark_scaling)
    ratio_sigmoid = np.minimum.accumulate(ratio_sigmoid[::-1])[::-1]
    next_max_size = max(
        np.searchsorted(ratio_sigmoid, (ratio_sigmoid[0] + 1) / 2, 'right') * 2,  # note, uses 1 instead of max, midpoint of sigmoid * 2
        np.searchsorted(ratio_sigmoid, minimum_saturation_ratio, 'right') * 3 // 2, # Explicitly finds minimum saturation ratio
    )
    token_change_too_small = next_max_size + start_x > (1 - minimum_token_change) * second_benchmark_points[-1]
    if token_change_too_small:
        return second_benchmark_points
    return generate_benchmark_points(start_x, next_max_size + start_x, points_ratio)


def get_benchmark_data_paddings(
    prev_benchmark_points: list[int], prev_benchmark_timings: list[int], new_benchmark_points: list[int], new_benchmark_timings: list[int], ratio, spline_k=3, spline_s=1) -> tuple[list[int], list[int]]:
    """Pad the new benchmark points and timings with the previous ones.

    This helps to ensure accurate interpolation when creating the batch timing estimators.
    """
    if prev_benchmark_points[-1] < new_benchmark_points[-1]:
        raise RuntimeError("Previous benchmark points must be less than or equal to new benchmark points.")
    if prev_benchmark_points[-1] == new_benchmark_points[-1]:
        return [], []  # No need to pad if the last points are the same
    spline_prev = UnivariateSpline(prev_benchmark_points, prev_benchmark_timings, k=spline_k, s=spline_s)
    # spline_new = UnivariateSpline(new_benchmark_points, new_benchmark_timings, k=spline_k, s=spline_s)
    # ratio: float = float(spline_new(new_benchmark_points[-1])) / float(spline_prev(new_benchmark_points[-1]))  # type: ignore
    if len(new_benchmark_points) < len(prev_benchmark_points):
        pad_benchmark_points = np.linspace(new_benchmark_points[-1], prev_benchmark_points[-1], 1 + len(prev_benchmark_points) - len(new_benchmark_points)).astype(int)[1:].tolist()
    else:
        pad_benchmark_points = [prev_benchmark_points[-1]]
    pad_timing_points = [int(spline_prev(point) * ratio) for point in pad_benchmark_points]  # type: ignore
    return pad_benchmark_points, pad_timing_points


def model_test_pattern_cpu(model_inferencer) -> int:
    """Run 3 loops with all combinations of gc.enable and time.sleep(0.1) and give minimum time."""
    timings: list[int] = []

    gc.collect()
    gc.freeze()
    gc.disable()

    def timedrun():
        timings.append(-time.perf_counter_ns())
        model_inferencer()
        timings[-1] += time.perf_counter_ns()

    # with torch.inference_mode(): toch inference mode should be in __init__
    # 3 runs, because 3rd time's the charm
    for count in range(3):
        # One run with gc, incase it works
        if count == 2:
            gc.enable()

        gc.collect()
        # Fresh run with pause and gc - [1, 1]
        timedrun()
        # Repeated runs - [0, 0]
        timedrun()
        timedrun()
        # Pause and run - [0, 1]
        time.sleep(0.1)
        timedrun()
        # Gc and run - [1, 0]
        gc.collect()
        timedrun()
        # Pause before fresh run
        time.sleep(0.1)

    return min(timings)  # Return minimum of timings (Best for CPU)


def model_test_pattern_gpu(model_inferencer) -> int:
    """Run 3 tests with a warmed up gpu and give median time."""
    timings = []

    gc.collect()
    gc.freeze()

    def timedrun():
        # sync and clear cache
        start_time = time.perf_counter_ns()
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()
        for runs in range(10):
            if time.perf_counter_ns() - start_time >= 2 and runs >= 3:  # atleast 3 times and 2 seconds
                break
            model_inferencer()
            # torch.cuda.synchronize()
        timings.append(-time.perf_counter_ns())
        model_inferencer()
        # torch.cuda.synchronize()
        timings[-1] += time.perf_counter_ns()

    # with torch.inference_mode(): torch inference mode should be in __init__
    # with torch.autocast("cuda"): should be in __init__
    for _ in range(3):
        time.sleep(0.1)
        timedrun()

    return sorted(timings)[len(timings) // 2]  # Return median of timings (Best for GPU)


# OLD VERSION for reference, not used in the current implementation and has worse performance
# def polynomial_interpolator_old(x_points, y_points, degree=2) -> np.ndarray:
#     """Create a 2nd order polynomial interpolator using numpy.polyfit"""
#     return Polynomial.fit(np.array(x_points, dtype=np.float64), np.array(y_points, dtype=np.float64), degree)(np.arange(x_points[-1] + 1, dtype=np.float64))


def spline_interpolator(x_points, y_points, max_tokens, min_tokens=0, smoothing=1, k=3) -> np.ndarray:
    """Create a spline interpolator using scipy.interpolate.UnivariateSpline."""
    spline = UnivariateSpline(x_points, y_points, s=smoothing, k=k)
    return np.asarray(spline(np.arange(x_points[-1] + 1, dtype=np.float64)))


def create_batch_timing_estimators(batch_benchmark_points: list[int], batch_timing_data: list[tuple[list[int], list[int]]], max_tokens: int, min_tokens: int = 0) -> np.ndarray:
    """Create array[batch_size, token_count] that gives estimated time for a batch of that size with that many tokens.

    min_tokens should be 0 unless we are specifically using an offset for the tokens. Not used in the current implementation.
    """
    batch_timing_estimators = []
    for batch_size in range(1, batch_benchmark_points[-1] + 1):
        if batch_size in batch_benchmark_points:
            found_idx = batch_benchmark_points.index(batch_size)
            token_points, timing_points = batch_timing_data[found_idx]
            interpolated_values = spline_interpolator(token_points, timing_points, max_tokens, min_tokens)
            batch_timing_estimators.append(interpolated_values)
        else:
            # If batch size is not in the benchmark points, we use interpolation between timings to estimate the points
            next_present_batch_size = next(filter(batch_size.__lt__, batch_benchmark_points))
            next_present_index = batch_benchmark_points.index(next_present_batch_size)
            prev_present_batch_size, prev_present_index = batch_benchmark_points[next_present_index - 1], next_present_index - 1
            prev_token_points, prev_timing_points = batch_timing_data[prev_present_index]
            next_token_points, next_timing_points = batch_timing_data[next_present_index]
            prev_interpolated_values = spline_interpolator(prev_token_points, prev_timing_points, max_tokens, min_tokens)
            next_interpolated_values = spline_interpolator(next_token_points, next_timing_points, max_tokens, min_tokens)
            # Linear interpolation between the two interpolated values
            ratio = (batch_size - prev_present_batch_size) / (next_present_batch_size - prev_present_batch_size)
            interpolated_values = prev_interpolated_values + ratio * (next_interpolated_values - prev_interpolated_values)
            batch_timing_estimators.append(interpolated_values)
    estimator = np.array(batch_timing_estimators, dtype=np.int64)
    assert not np.any(np.isnan(estimator)), "NaN values found in the estimator"
    assert not np.any(np.isinf(estimator)), "Inf values found in the estimator"
    assert not np.any(estimator <= 0), "Negative or zero values found in the estimator"
    return estimator
