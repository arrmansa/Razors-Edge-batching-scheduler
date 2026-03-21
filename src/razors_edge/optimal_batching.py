"""Optimal batching algorithms for efficient model inference with dynamic programming and RMS latency optimization."""

import numpy as np
from numba import njit


@njit(fastmath=True, nogil=True, boundscheck=False)
def _compiled_dynamic_batcher(sorted_model_input_sizes: np.ndarray, batch_timing_estimators: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dynamic programming approach to compute minimum time needed to process all model_inputs in batches."""
    n = len(sorted_model_input_sizes)
    prev_batch_size = np.zeros(n + 1, dtype=np.uint8)
    batch_start_times = np.zeros(n + 1, dtype=np.int64) + np.iinfo(np.int64).max
    batch_start_times[0] = 0
    for batch_start_idx in range(n):
        for estimator_idx in range(len(batch_timing_estimators)):
            batch_size = estimator_idx + 1
            batch_end_idx = batch_start_idx + estimator_idx
            next_batch_start_idx = batch_end_idx + 1
            if batch_end_idx >= n:
                continue
            last_model_input_size = sorted_model_input_sizes[batch_end_idx]
            next_batch_start_time = batch_timing_estimators[estimator_idx, last_model_input_size] + batch_start_times[batch_start_idx]
            if next_batch_start_time < batch_start_times[next_batch_start_idx]:
                batch_start_times[next_batch_start_idx] = next_batch_start_time
                prev_batch_size[next_batch_start_idx] = batch_size
    return batch_start_times, prev_batch_size


# math.log2 is 2x slower than binary search for bit length.
# @njit(fastmath=True, nogil=True, boundscheck=False)
# def compiled_bit_length_v3(x: int) -> int:
#     return math.ceil(math.log2(x))+1


@njit(fastmath=True, nogil=True, boundscheck=False)
def _compiled_bit_length(x: int) -> int:
    """Calculate bit length using efficient bit manipulation tricks.

    0 is not handled, as it is not a valid input for this function.
    """
    # Use binary search approach - much faster for large numbers
    bit_length = 0
    # In theory I could remove the 32
    if x >= 1 << 32:
        bit_length += 32
        x >>= 32
    if x >= 1 << 16:
        bit_length += 16
        x >>= 16
    if x >= 1 << 8:
        bit_length += 8
        x >>= 8
    if x >= 1 << 4:
        bit_length += 4
        x >>= 4
    if x >= 1 << 2:
        bit_length += 2
        x >>= 2
    if x >= 1 << 1:
        bit_length += 1
    if x >= 1 << 0:
        bit_length += 1
    return bit_length


@njit(fastmath=True, nogil=True, boundscheck=False)
def _prospective_rms_latency_improvement(
    chosen_process_duration: int,
    chosen_mean_queueing_time: int,
    chosen_batch_size: int,
    prospective_process_duration: int,
    prospective_mean_queueing_time: int,
    prospective_batch_size: int,
) -> bool:
    """Determine if scheduling the prospective batch will result in lower RMS latency compared to chosen batch.
    
    EQUIVALENT TO: 
    
    mean_square_latency_1 = chosen_batch_size * ((chosen_mean_queueing_time + chosen_process_duration) ** 2) + \
    prospective_batch_size * ((prospective_mean_queueing_time  + chosen_process_duration + prospective_process_duration) ** 2)
    
    mean_square_latency_2 = prospective_batch_size * ((prospective_mean_queueing_time + prospective_process_duration) ** 2) + \
    chosen_batch_size * ((chosen_mean_queueing_time + prospective_process_duration + chosen_process_duration) ** 2)

    return mean_square_latency_1 > mean_square_latency_2
    # But we can simplify this to a more efficient comparison that avoids overflow issues:
    """
    # Maximum happy case is 256 * x * (2.5x) < 2^63 - 1
    # x can have 26 bits of precision since log2((2**63 / (256 * 5)) ** 0.5) = 26.8247...
    # New dynamic bitshift logic to avoid overflow in multiplication, full 64 bit support
    MAX_BIT_LENGTH = 26
    process_duration_bitshift = max(_compiled_bit_length(max(chosen_process_duration, prospective_process_duration)) - MAX_BIT_LENGTH, 0)
    queueing_time_bitshift = max(_compiled_bit_length(max(chosen_mean_queueing_time, prospective_mean_queueing_time)) - MAX_BIT_LENGTH, 0)
    # Scale down the durations and queueing times to avoid overflow in multiplication
    chosen_process_duration >>= process_duration_bitshift
    prospective_process_duration >>= process_duration_bitshift
    chosen_mean_queueing_time >>= queueing_time_bitshift
    prospective_mean_queueing_time >>= queueing_time_bitshift
    # The bit shifts ensure that we do not overflow when multiplying the batch sizes with the durations and queueing times.

    # OLD VERSION, very minimal performance gain, but worse overflow safety, kept for reference
    # OVERFLOW_SAFETY_SHIFT = 16
    # A fixed scale down of 2^16 for multiplication overflow safety, resolution drops to ~65us
    # Maximum happy case is 256 * x * (5x) < (2^63 - 1) * 2^16 * 2^16
    # 1.5 hours of both queue wait and duration
    # chosen_process_duration >>= OVERFLOW_SAFETY_SHIFT
    # prospective_process_duration >>= OVERFLOW_SAFETY_SHIFT
    # chosen_mean_queueing_time >>= OVERFLOW_SAFETY_SHIFT
    # prospective_mean_queueing_time >>= OVERFLOW_SAFETY_SHIFT

    # Compare the two RMS latency calculations
    return chosen_batch_size * prospective_process_duration * (chosen_mean_queueing_time + chosen_process_duration + prospective_mean_queueing_time // 2) \
        < prospective_batch_size * chosen_process_duration * (prospective_mean_queueing_time + prospective_process_duration + chosen_mean_queueing_time // 2)


@njit(fastmath=True, nogil=True, boundscheck=False)
def _get_slice_indexes_and_duration(prev_sizes_array: np.ndarray, batch_start_times: np.ndarray, queueing_times_array: np.ndarray) -> tuple[int, int, int]:
    """Get optimal batch slice indexes and duration using RMS latency comparison."""
    chosen_end_idx: int = len(prev_sizes_array) - 1  # Sane defaults
    chosen_start_idx: int = chosen_end_idx - prev_sizes_array[chosen_end_idx]
    chosen_process_duration: int = batch_start_times[chosen_end_idx] - batch_start_times[chosen_start_idx]
    chosen_batch_size: int = chosen_end_idx - chosen_start_idx
    chosen_mean_queueing_time: int = queueing_times_array[chosen_start_idx:chosen_end_idx].sum() // chosen_batch_size
    end_idx: int = chosen_start_idx
    while end_idx > 0:
        start_idx = end_idx - prev_sizes_array[end_idx]
        prospective_process_duration: int = batch_start_times[end_idx] - batch_start_times[start_idx]
        prospective_batch_size: int = end_idx - start_idx
        prospective_mean_queueing_time: int = queueing_times_array[start_idx:end_idx].sum() // prospective_batch_size

        # RMS LATENCY COMPARISON
        if _prospective_rms_latency_improvement(
            chosen_process_duration,
            chosen_mean_queueing_time,
            chosen_batch_size,
            prospective_process_duration,
            prospective_mean_queueing_time,
            prospective_batch_size,
        ):
            # If the prospective batch is better, update the chosen batch
            chosen_start_idx = start_idx
            chosen_end_idx = end_idx
            chosen_process_duration = prospective_process_duration
            chosen_mean_queueing_time = prospective_mean_queueing_time
            chosen_batch_size = prospective_batch_size

        end_idx = start_idx

    return chosen_start_idx, chosen_end_idx, chosen_process_duration


def get_batch_start_end_idx_and_duration(
    sorted_model_input_sizes: tuple[int] | np.ndarray,
    batch_timing_estimators: np.ndarray,
    queuing_times: tuple[int] | np.ndarray,
    expected_schedule_time: int,
) -> tuple[int, int, int]:
    """Get optimal batch start and end indexes with duration using dynamic programming and RMS latency optimization."""
    batch_start_times, prev_batch_size = _compiled_dynamic_batcher(np.asarray(sorted_model_input_sizes, dtype=np.uint32), batch_timing_estimators)
    queueing_times_array = np.clip(expected_schedule_time - np.asarray(queuing_times, dtype=np.int64), 0, None)
    return _get_slice_indexes_and_duration(prev_batch_size, batch_start_times, queueing_times_array)


# To trigger the JIT compilation, with realistic parameters
# estimators = np.array(
#     [[int((model_input_per_str**1.5) * (string_count**0.5)) for model_input_per_str in range(1000 + 1)] for string_count in range(1, 17)],
#     dtype=np.int64,
# )
# output = get_batch_start_end_idx_and_duration(tuple(np.linspace(10, 1000, 32, dtype=np.uint32).tolist()), estimators, tuple(np.arange(32).tolist()), 1000)
# print(f"Got output {output}")
