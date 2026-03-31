from typing import Literal

from numba import njit
import numpy as np


@njit(fastmath=True, nogil=True, boundscheck=False)
def _compiled_greedy_batcher_lookahead1(
    sorted_model_input_sizes: np.ndarray,
    batch_timing_estimators: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lookahead-1 greedy batching:
    At each step, compare:
        - extend current batch
        - split and start new batch
    """

    n = len(sorted_model_input_sizes)
    max_batch_size = len(batch_timing_estimators)

    prev_batch_size = np.zeros(n + 1, dtype=np.uint8)
    batch_start_times = np.zeros(n + 1, dtype=np.int64)

    batch_start_times[0] = 0

    i = 0
    while i < n:
        j = i
        current_max = sorted_model_input_sizes[i]

        # grow batch greedily
        while j + 1 < n:
            next_size = sorted_model_input_sizes[j + 1]
            next_max = next_size  # sorted, so it's always increasing

            batch_size = j - i + 1
            if batch_size >= max_batch_size:
                break

            # ---- cost if we extend ----
            # estimator index = batch_size (since +1)
            c_extend = batch_timing_estimators[batch_size, next_max]

            # ---- cost if we split ----
            # current batch cost
            c_current = batch_timing_estimators[batch_size - 1, current_max]

            # next item alone
            c_single = batch_timing_estimators[0, next_max]

            c_split = c_current + c_single

            if c_extend <= c_split:
                j += 1
                current_max = next_max
            else:
                break

        # finalize batch [i, j]
        batch_size = j - i + 1
        end_idx = j + 1

        batch_cost = batch_timing_estimators[batch_size - 1, current_max]

        batch_start_times[end_idx] = batch_start_times[i] + batch_cost
        prev_batch_size[end_idx] = batch_size

        i = j + 1

    return batch_start_times, prev_batch_size




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
        if prospective_max_queueing_time + prospective_process_duration > chosen_max_queueing_time + chosen_process_duration:
            chosen_start_idx = start_idx
            chosen_end_idx = end_idx
            chosen_process_duration = prospective_process_duration
            chosen_max_queueing_time = prospective_max_queueing_time
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
    strategy: Literal['MINMAX', 'FIFO', 'GUARDED_BATCH_SIZE'],
) -> tuple[int, int, int]:
    """Get optimal batch start and end indexes with duration using dynamic programming and latency optimization."""
    sorted_model_input_sizes = np.asarray(sorted_model_input_sizes, dtype=np.uint32)
    batch_start_times, prev_batch_size = _compiled_greedy_batcher_lookahead1(sorted_model_input_sizes, batch_timing_estimators)
    queueing_times_array = np.clip(expected_schedule_time - np.asarray(queuing_times, dtype=np.int64), 0, None)
    if strategy == 'MINMAX':
        return _get_slice_indexes_and_duration_minmax(prev_batch_size, batch_start_times, queueing_times_array)
    elif strategy == 'FIFO':
        return _get_slice_indexes_and_duration_fifo(prev_batch_size, batch_start_times, queueing_times_array)
    elif strategy == 'GUARDED_BATCH_SIZE':
        return _get_slice_indexes_and_duration_guarded_batch_size(prev_batch_size, batch_start_times, queueing_times_array)
    else:
        raise RuntimeError("Invalid Strategy")
