"""Optimal batching algorithms for efficient model inference with dynamic programming and RMS latency optimization."""

from typing import Literal

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
def _get_slice_indexes_and_duration_rms(prev_sizes_array: np.ndarray, batch_start_times: np.ndarray, queueing_times_array: np.ndarray) -> tuple[int, int, int]:
    """Get optimal batch slice indexes and duration using RMS latency comparison."""
    chosen_end_idx: int = len(prev_sizes_array) - 1
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


@njit(fastmath=True, nogil=True, boundscheck=False)
def _get_slice_indexes_and_duration_fifo(prev_sizes_array: np.ndarray, batch_start_times: np.ndarray, queueing_times_array: np.ndarray) -> tuple[int, int, int]:
    """Get optimal batch slice indexes and duration using fifo."""
    chosen_end_idx: int = len(prev_sizes_array) - 1
    chosen_start_idx: int = chosen_end_idx - prev_sizes_array[chosen_end_idx]
    chosen_process_duration: int = batch_start_times[chosen_end_idx] - batch_start_times[chosen_start_idx]
    chosen_max_queueing_time: int = queueing_times_array[chosen_start_idx:chosen_end_idx].max()
    end_idx: int = chosen_start_idx
    while end_idx > 0:
        start_idx = end_idx - prev_sizes_array[end_idx]
        prospective_process_duration: int = batch_start_times[end_idx] - batch_start_times[start_idx]
        prospective_max_queueing_time: int = queueing_times_array[start_idx:end_idx].max()

        # Batch with element that came first is chosen
        if prospective_max_queueing_time > chosen_max_queueing_time:
            chosen_start_idx = start_idx
            chosen_end_idx = end_idx
            chosen_process_duration = prospective_process_duration
            chosen_max_queueing_time = prospective_max_queueing_time

        end_idx = start_idx

    return chosen_start_idx, chosen_end_idx, chosen_process_duration


@njit(fastmath=True, nogil=True, boundscheck=False)
def _get_slice_indexes_and_duration_minmax(prev_sizes_array: np.ndarray, batch_start_times: np.ndarray, queueing_times_array: np.ndarray) -> tuple[int, int, int]:
    """Get optimal batch slice indexes and duration by minimizing max latency."""
    chosen_end_idx: int = len(prev_sizes_array) - 1
    chosen_start_idx: int = chosen_end_idx - prev_sizes_array[chosen_end_idx]
    chosen_process_duration: int = batch_start_times[chosen_end_idx] - batch_start_times[chosen_start_idx]
    chosen_max_queueing_time: int = queueing_times_array[chosen_start_idx:chosen_end_idx].max()
    end_idx: int = chosen_start_idx
    while end_idx > 0:
        start_idx = end_idx - prev_sizes_array[end_idx]
        prospective_process_duration: int = batch_start_times[end_idx] - batch_start_times[start_idx]
        prospective_max_queueing_time: int = queueing_times_array[start_idx:end_idx].max()

        # Batch with element that will have the largest latency is chosen
        if ((prospective_max_queueing_time>>1) + (prospective_process_duration>>1)) > ((chosen_max_queueing_time>>1) + (chosen_process_duration>>1)):
            chosen_start_idx = start_idx
            chosen_end_idx = end_idx
            chosen_process_duration = prospective_process_duration
            chosen_max_queueing_time = prospective_max_queueing_time

        end_idx = start_idx

    return chosen_start_idx, chosen_end_idx, chosen_process_duration


@njit(fastmath=True, nogil=True, boundscheck=False)
def _get_slice_indexes_and_duration_meanmax(prev_sizes_array: np.ndarray, batch_start_times: np.ndarray, queueing_times_array: np.ndarray) -> tuple[int, int, int]:
    """Get optimal batch slice indexes and duration by minimizing max latency."""
    chosen_end_idx: int = len(prev_sizes_array) - 1
    chosen_start_idx: int = chosen_end_idx - prev_sizes_array[chosen_end_idx]
    chosen_process_duration: int = batch_start_times[chosen_end_idx] - batch_start_times[chosen_start_idx]
    chosen_mean_queueing_time: int = queueing_times_array[chosen_start_idx:chosen_end_idx].mean()
    end_idx: int = chosen_start_idx
    while end_idx > 0:
        start_idx = end_idx - prev_sizes_array[end_idx]
        prospective_process_duration: int = batch_start_times[end_idx] - batch_start_times[start_idx]
        prospective_mean_queueing_time: int = queueing_times_array[start_idx:end_idx].mean()

        # Batch with element that will have the largest latency is chosen
        if prospective_mean_queueing_time + prospective_process_duration > chosen_mean_queueing_time + chosen_process_duration:
            chosen_start_idx = start_idx
            chosen_end_idx = end_idx
            chosen_process_duration = prospective_process_duration
            chosen_mean_queueing_time = prospective_mean_queueing_time

        end_idx = start_idx

    return chosen_start_idx, chosen_end_idx, chosen_process_duration



@njit(fastmath=True, nogil=True, boundscheck=False)
def _get_slice_indexes_and_duration_efficiency(prev_sizes_array: np.ndarray, batch_start_times: np.ndarray, queueing_times_array: np.ndarray, input_timings_array: np.ndarray) -> tuple[int, int, int]:
    """Get optimal batch slice indexes and duration by choosing the batch with most efficiency which has one element older than mean queuing time."""
    mean_queueing_time = queueing_times_array.sum() // len(queueing_times_array)
    chosen_end_idx: int = len(prev_sizes_array) - 1
    chosen_start_idx: int = chosen_end_idx - prev_sizes_array[chosen_end_idx]
    chosen_process_duration: int = batch_start_times[chosen_end_idx] - batch_start_times[chosen_start_idx]
    chosen_efficiency: float = input_timings_array[chosen_start_idx:chosen_end_idx].sum() / chosen_process_duration
    end_idx: int = chosen_start_idx
    while end_idx > 0:
        start_idx = end_idx - prev_sizes_array[end_idx]
        if queueing_times_array[start_idx:end_idx].max() >= mean_queueing_time:
                
            prospective_process_duration: int = batch_start_times[end_idx] - batch_start_times[start_idx]
            prospective_efficiency: float = input_timings_array[start_idx:end_idx].sum() / prospective_process_duration

            if prospective_efficiency >= chosen_efficiency:
                chosen_start_idx = start_idx
                chosen_end_idx = end_idx
                chosen_process_duration = prospective_process_duration
                chosen_efficiency = prospective_efficiency

        end_idx = start_idx

    return chosen_start_idx, chosen_end_idx, chosen_process_duration


# @njit(fastmath=True, nogil=True, boundscheck=False)
# def _get_slice_indexes_and_duration_efficiency(prev_sizes_array: np.ndarray, batch_start_times: np.ndarray, queueing_times_array: np.ndarray) -> tuple[int, int, int]:
#     """Get optimal batch slice indexes and duration by choosing the batch with most efficiency which has one element older than mean queuing time."""
#     mean_queueing_time = queueing_times_array.sum() // len(queueing_times_array)
#     chosen_end_idx: int = len(prev_sizes_array) - 1
#     chosen_start_idx: int = chosen_end_idx - prev_sizes_array[chosen_end_idx]
#     chosen_process_duration: int = batch_start_times[chosen_end_idx] - batch_start_times[chosen_start_idx]
#     chosen_size: float = chosen_start_idx - chosen_end_idx
#     end_idx: int = chosen_start_idx
#     while end_idx > 0:
#         start_idx = end_idx - prev_sizes_array[end_idx]
#         if queueing_times_array[start_idx:end_idx].max() >= mean_queueing_time:
                
#             prospective_process_duration: int = batch_start_times[end_idx] - batch_start_times[start_idx]
#             prospective_size: float = start_idx - end_idx

#             if prospective_size > chosen_size:
#                 chosen_start_idx = start_idx
#                 chosen_end_idx = end_idx
#                 chosen_process_duration = prospective_process_duration
#                 chosen_size = prospective_size

#         end_idx = start_idx

#     return chosen_start_idx, chosen_end_idx, chosen_process_duration


@njit(fastmath=True, nogil=True, boundscheck=False)
def _get_slice_indexes_and_duration_srpt_aging(prev_sizes_array: np.ndarray, batch_start_times: np.ndarray, queueing_times_array: np.ndarray) -> tuple[int, int, int]:
    """Pick shortest processing time batches while applying an aging factor."""
    mean_queueing_time = queueing_times_array.sum() // len(queueing_times_array)
    max_queueing_time = queueing_times_array.max()
    waiting_pressure = 1 + (max_queueing_time // max(1, mean_queueing_time + 1))

    chosen_end_idx: int = len(prev_sizes_array) - 1
    chosen_start_idx: int = chosen_end_idx - prev_sizes_array[chosen_end_idx]
    chosen_process_duration: int = batch_start_times[chosen_end_idx] - batch_start_times[chosen_start_idx]
    chosen_batch_size: int = chosen_end_idx - chosen_start_idx
    chosen_score: int = chosen_process_duration * 1000 // max(1, chosen_batch_size * waiting_pressure)

    end_idx: int = chosen_start_idx
    while end_idx > 0:
        start_idx = end_idx - prev_sizes_array[end_idx]
        prospective_process_duration: int = batch_start_times[end_idx] - batch_start_times[start_idx]
        prospective_batch_size: int = end_idx - start_idx
        prospective_score: int = prospective_process_duration * 1000 // max(1, prospective_batch_size * waiting_pressure)

        if prospective_score < chosen_score:
            chosen_start_idx = start_idx
            chosen_end_idx = end_idx
            chosen_process_duration = prospective_process_duration
            chosen_batch_size = prospective_batch_size
            chosen_score = prospective_score
        end_idx = start_idx

    return chosen_start_idx, chosen_end_idx, chosen_process_duration


@njit(fastmath=True, nogil=True, boundscheck=False)
def _get_slice_indexes_and_duration_hrrn(prev_sizes_array: np.ndarray, batch_start_times: np.ndarray, queueing_times_array: np.ndarray) -> tuple[int, int, int]:
    """Highest response ratio next score at the batch level."""
    chosen_end_idx: int = len(prev_sizes_array) - 1
    chosen_start_idx: int = chosen_end_idx - prev_sizes_array[chosen_end_idx]
    chosen_process_duration: int = batch_start_times[chosen_end_idx] - batch_start_times[chosen_start_idx]
    chosen_batch_size: int = chosen_end_idx - chosen_start_idx
    chosen_mean_queueing_time: int = queueing_times_array[chosen_start_idx:chosen_end_idx].sum() // max(1, chosen_batch_size)
    chosen_score: int = ((chosen_mean_queueing_time + chosen_process_duration) * 1000) // max(1, chosen_process_duration)

    end_idx: int = chosen_start_idx
    while end_idx > 0:
        start_idx = end_idx - prev_sizes_array[end_idx]
        prospective_process_duration: int = batch_start_times[end_idx] - batch_start_times[start_idx]
        prospective_batch_size: int = end_idx - start_idx
        prospective_mean_queueing_time: int = queueing_times_array[start_idx:end_idx].sum() // max(1, prospective_batch_size)
        prospective_score: int = ((prospective_mean_queueing_time + prospective_process_duration) * 1000) // max(1, prospective_process_duration)

        if prospective_score > chosen_score:
            chosen_start_idx = start_idx
            chosen_end_idx = end_idx
            chosen_process_duration = prospective_process_duration
            chosen_score = prospective_score
        end_idx = start_idx

    return chosen_start_idx, chosen_end_idx, chosen_process_duration


@njit(fastmath=True, nogil=True, boundscheck=False)
def _get_slice_indexes_and_duration_efficiency(prev_sizes_array: np.ndarray, batch_start_times: np.ndarray, queueing_times_array: np.ndarray) -> tuple[int, int, int]:
    """Get optimal batch slice indexes and duration using RMS latency comparison."""


@njit(fastmath=True, nogil=True, boundscheck=False)
def _get_slice_indexes_and_duration_throughput_aging(prev_sizes_array: np.ndarray, batch_start_times: np.ndarray, queueing_times_array: np.ndarray) -> tuple[int, int, int]:
    """Maximize throughput while adding mild pressure to flush aged queue entries."""
    mean_queueing_time = queueing_times_array.sum() // len(queueing_times_array)
    chosen_end_idx: int = len(prev_sizes_array) - 1
    chosen_start_idx: int = chosen_end_idx - prev_sizes_array[chosen_end_idx]
    chosen_process_duration: int = batch_start_times[chosen_end_idx] - batch_start_times[chosen_start_idx]
    chosen_batch_size: int = chosen_end_idx - chosen_start_idx
    chosen_max_queueing_time: int = queueing_times_array[chosen_start_idx:chosen_end_idx].max()
    chosen_score: int = (chosen_batch_size * 1_000_000) // max(1, chosen_process_duration) + (chosen_max_queueing_time // max(1, mean_queueing_time + 1))
    end_idx: int = chosen_start_idx
    while end_idx > 0:
        start_idx = end_idx - prev_sizes_array[end_idx]
        prospective_process_duration: int = batch_start_times[end_idx] - batch_start_times[start_idx]
        prospective_batch_size: int = end_idx - start_idx
        prospective_max_queueing_time: int = queueing_times_array[start_idx:end_idx].max()
        prospective_score: int = (prospective_batch_size * 1_000_000) // max(1, prospective_process_duration) + (
            prospective_max_queueing_time // max(1, mean_queueing_time + 1)
        )
        if prospective_score > chosen_score:
            chosen_start_idx = start_idx
            chosen_end_idx = end_idx
            chosen_process_duration = prospective_process_duration
            chosen_score = prospective_score
        end_idx = start_idx
    return chosen_start_idx, chosen_end_idx, chosen_process_duration

@njit(fastmath=True, nogil=True, boundscheck=False)
def _get_slice_indexes_and_duration_minmax_guarded_rmse(prev_sizes_array: np.ndarray, batch_start_times: np.ndarray, queueing_times_array: np.ndarray) -> tuple[int, int, int]:
    """Pick candidates that beat MINMAX guardrails, then choose the lowest RMSE proxy."""
    mean_minmax_time = (queueing_times_array.sum() + batch_start_times[-1]) // len(queueing_times_array)
    chosen_end_idx: int = len(prev_sizes_array) - 1
    chosen_start_idx: int = chosen_end_idx - prev_sizes_array[chosen_end_idx]
    chosen_process_duration: int = batch_start_times[chosen_end_idx] - batch_start_times[chosen_start_idx]
    chosen_batch_size: int = chosen_end_idx - chosen_start_idx
    chosen_mean_queueing_time: int = queueing_times_array[chosen_start_idx:chosen_end_idx].sum() // chosen_batch_size
    end_idx: int = chosen_start_idx
    while end_idx > 0:
        start_idx = end_idx - prev_sizes_array[end_idx]
        prospective_process_duration: int = batch_start_times[end_idx] - batch_start_times[start_idx]
        if queueing_times_array[start_idx:end_idx].max() + prospective_process_duration >= mean_minmax_time:

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


@njit(fastmath=True, nogil=True, boundscheck=False)
def _get_slice_indexes_and_duration_efficiency(prev_sizes_array: np.ndarray, batch_start_times: np.ndarray, queueing_times_array: np.ndarray, input_timings_array: np.ndarray) -> tuple[int, int, int]:
    """Get optimal batch slice indexes and duration by maximizing efficiency (based on input_timings_array)."""
    mean_minmax_time = (queueing_times_array.sum() + batch_start_times[-1]) // len(queueing_times_array)
    chosen_end_idx: int = len(prev_sizes_array) - 1
    chosen_start_idx: int = chosen_end_idx - prev_sizes_array[chosen_end_idx]
    chosen_process_duration: float = batch_start_times[chosen_end_idx] - batch_start_times[chosen_start_idx]
    chosen_efficiency: float = input_timings_array[chosen_start_idx:chosen_end_idx].sum() / chosen_process_duration
    end_idx: int = chosen_start_idx
    while end_idx > 0:
        start_idx = end_idx - prev_sizes_array[end_idx]
        prospective_process_duration: int = batch_start_times[end_idx] - batch_start_times[start_idx]
        if queueing_times_array[start_idx:end_idx].max() + prospective_process_duration >= mean_minmax_time:
            prospective_efficiency: int = input_timings_array[start_idx:end_idx].sum() / prospective_process_duration

            # Batch with element that will have the largest latency is chosen
            if prospective_efficiency > chosen_efficiency:
                chosen_start_idx = start_idx
                chosen_end_idx = end_idx
                chosen_process_duration = prospective_process_duration
                chosen_efficiency = prospective_efficiency
        end_idx = start_idx

    return chosen_start_idx, chosen_end_idx, chosen_process_duration




@njit(fastmath=True, nogil=True, boundscheck=False)
def _get_slice_indexes_and_duration_guarded_batch_size(prev_sizes_array: np.ndarray, batch_start_times: np.ndarray, queueing_times_array: np.ndarray) -> tuple[int, int, int]:
    """Pick candidates that beats mean MINMAX guardrails, then choose the largest batch size."""
    mean_minmax_time = (queueing_times_array.sum() + batch_start_times[-1]) // len(queueing_times_array)
    chosen_end_idx: int = len(prev_sizes_array) - 1
    chosen_start_idx: int = chosen_end_idx - prev_sizes_array[chosen_end_idx]
    chosen_process_duration: int = batch_start_times[chosen_end_idx] - batch_start_times[chosen_start_idx]
    chosen_batch_size: int = chosen_end_idx - chosen_start_idx
    chosen_minmax_time: int = chosen_process_duration + queueing_times_array[chosen_start_idx: chosen_end_idx].max()
    end_idx: int = chosen_start_idx
    while end_idx > 0:
        start_idx = end_idx - prev_sizes_array[end_idx]
        prospective_process_duration: int = batch_start_times[end_idx] - batch_start_times[start_idx]
        prospective_minmax_time: int = prospective_process_duration + queueing_times_array[start_idx:end_idx].max()
        if prospective_minmax_time >= mean_minmax_time:
            prospective_batch_size: int = end_idx - start_idx
            # RMS LATENCY COMPARISON
            if prospective_batch_size > chosen_batch_size or (prospective_batch_size == chosen_batch_size and prospective_minmax_time > chosen_minmax_time):
                # If the prospective batch is better, update the chosen batch
                chosen_start_idx = start_idx
                chosen_end_idx = end_idx
                chosen_process_duration = prospective_process_duration
                chosen_batch_size = prospective_batch_size
                chosen_minmax_time = prospective_minmax_time
        end_idx = start_idx

    return chosen_start_idx, chosen_end_idx, chosen_process_duration


def get_batch_start_end_idx_and_duration(
    sorted_model_input_sizes: tuple[int, ...] | np.ndarray,
    batch_timing_estimators: np.ndarray,
    queuing_times: tuple[int, ...] | np.ndarray,
    expected_schedule_time: int,
    strategy: Literal['RMS', 'FIFO', 'MINMAX', 'EFFICIENCY', 'SRPT_AGING', 'HRRN', 'THROUGHPUT_AGING', 'BALANCED_TUNED', 'MINMAX_GUARDED_RMSE'],
) -> tuple[int, int, int]:
    """Get optimal batch start and end indexes with duration using dynamic programming and RMS latency optimization."""
    sorted_model_input_sizes = np.asarray(sorted_model_input_sizes, dtype=np.uint32)
    batch_start_times, prev_batch_size = _compiled_dynamic_batcher(sorted_model_input_sizes, batch_timing_estimators)
    queueing_times_array = np.clip(expected_schedule_time - np.asarray(queuing_times, dtype=np.int64), 0, None)
    if strategy == 'RMS':
        return _get_slice_indexes_and_duration_rms(prev_batch_size, batch_start_times, queueing_times_array)
    elif strategy == 'FIFO':
        return _get_slice_indexes_and_duration_fifo(prev_batch_size, batch_start_times, queueing_times_array)
    elif strategy == 'MINMAX':
        return _get_slice_indexes_and_duration_minmax(prev_batch_size, batch_start_times, queueing_times_array)
    elif strategy == 'MEANMAX':
        return _get_slice_indexes_and_duration_meanmax(prev_batch_size, batch_start_times, queueing_times_array)
    elif strategy == 'EFFICIENCY':
        return _get_slice_indexes_and_duration_efficiency(prev_batch_size, batch_start_times, queueing_times_array, batch_timing_estimators[0][sorted_model_input_sizes-1])
    elif strategy == 'SRPT_AGING':
        return _get_slice_indexes_and_duration_srpt_aging(prev_batch_size, batch_start_times, queueing_times_array)
    elif strategy == 'HRRN':
        return _get_slice_indexes_and_duration_hrrn(prev_batch_size, batch_start_times, queueing_times_array)
    elif strategy == 'THROUGHPUT_AGING':
        return _get_slice_indexes_and_duration_throughput_aging(prev_batch_size, batch_start_times, queueing_times_array)
    elif strategy == 'MINMAX_GUARDED_RMSE':
        return _get_slice_indexes_and_duration_minmax_guarded_rmse(prev_batch_size, batch_start_times, queueing_times_array)
    elif strategy == 'BATCH_SIZE':
        return _get_slice_indexes_and_duration_guarded_batch_size(prev_batch_size, batch_start_times, queueing_times_array)
    else:
        raise RuntimeError("Invalid Strategy")
