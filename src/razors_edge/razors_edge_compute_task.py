"""Abstract base class for Razors Edge compute tasks."""

import logging
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, override, Literal
from functools import partial
from src.executor.base_batched_compute_task import BaseBatchedComputeTask


class RazorsEdgeComputeTask(BaseBatchedComputeTask):
    """
    Base class for any compute tasks batched with razors edge that needs to run.
    Has improved batching logic based on 
    1. loading and initializing models (user defined)
    2. Automatically generating test inputs based on batch size and token count
    3. Automatically benchmarking the model time with test inputs
    4. Automatically choosing the best batch to run inference on
    5. User defined post processing
    This also avoids loading libraries like numpy, scipy and other razors edge functionality just by importing this class
    """

    _cache_for_batch_timing_estimators = {}

    @staticmethod
    @override
    def get_batch_start_end_idx_and_duration(sorted_model_input_sizes: tuple[int], batch_timing_estimators, queuing_times: tuple[int], expected_schedule_time: int) -> tuple[int, int, int]:
        raise NotImplementedError("This method should not be called, use the one from optimal_batching instead.")

    @property
    @override
    def batch_benchmark_sizes(self) -> list[int]:
        """Batch sizes for benchmarking upto max batch size"""

    @property
    @override
    def min_input_size(self) -> int:
        """Min size of model input"""

    @property
    @override
    def max_input_size(self) -> int:
        """Max size of model input"""

    @property
    @override
    def max_input_points(self) -> int:    
        """Max number of input points to consider for a given batch benchmark"""

    @property
    @override
    def is_gpu(self) -> bool:
        """Return whether this task runs GPU inference."""

    @property
    def latency_strategy(self) -> Literal['FIFO', 'MINMAX', 'GUARDED_BATCH_SIZE']:
        return 'MINMAX'
    
    @property
    def enable_cache_batch_timing_estimators(self) -> bool:
        return True

    @classmethod
    @override
    def model_test_pattern(cls, model_inferencer) -> int:
        """Benchmark a callable inference workload."""
        from src.razors_edge.optimal_benchmarking import model_test_pattern_cpu, model_test_pattern_gpu
        if cls.is_gpu:
            return model_test_pattern_gpu(model_inferencer)
        else:
            return model_test_pattern_cpu(model_inferencer)

    @override
    def get_input_size(self, input_data: Any, preprocessed_input: Any) -> int:
        """Gives the integer size of model input"""

    @override
    def generate_test_input(self, batch_size: int, input_size: int) -> tuple[tuple, dict]:
        """create a test input for the model"""

    @override
    def load_model(self, model_pool: ThreadPoolExecutor) -> Any:
        """load and initialize model, tokenizer, buffers, etc."""

    def _find_load_model_provider(self):
        """
        Find the nearest class in MRO that actually defines `load_model`
        And doesn't just inherit it from RazorsEdgeComputeTask
        """
        for cls in type(self).__mro__:
            if "load_model" in cls.__dict__:
                return cls
        raise RuntimeError("No helper implementation found")

    @override
    def batch_inference_times(self, multiple_model_inputs: Sequence[tuple[tuple, dict]]):
        """Measure inference times for one or more texts."""
        return [self.model_test_pattern(partial(self.model, *args, **kwargs)) for args, kwargs in multiple_model_inputs]

    def get_batch_timing_data(self):
        """Generate batch timing data for various batch sizes and model input sizes."""
        from src.razors_edge.optimal_benchmarking import calculate_next_benchmark_points, generate_benchmark_points, get_benchmark_data_paddings, get_points_ratio

        points_ratio = get_points_ratio(self.min_input_size, self.max_input_size, self.max_input_points)
        initial_input_size_benchmark_points = generate_benchmark_points(self.min_input_size, self.max_input_size, points_ratio)
        first_benchmarking_model_inputs = list(map(partial(self.generate_test_input, 1), initial_input_size_benchmark_points))
        second_benchmarking_model_inputs = list(map(partial(self.generate_test_input, 2), initial_input_size_benchmark_points))
        # Warmup with the highest input size and batch size
        warmup_data: list[int] = self.batch_inference_times([self.generate_test_input(self.batch_benchmark_sizes[-1], self.max_input_size)])
        logging.warning(f"Warmup times: {warmup_data}")

        # Note: Benchmarking from input size highest to lowest after warmup which minimizes the noise in timings.
        # Need first 2 points to switch to the fast benchmarking method.
        batch_timing_data = [
            (initial_input_size_benchmark_points, list(reversed(self.batch_inference_times(reversed(first_benchmarking_model_inputs))))),
            (initial_input_size_benchmark_points, list(reversed(self.batch_inference_times(reversed(second_benchmarking_model_inputs))))),
        ]
        logging.warning(f"Batch 1 times: {batch_timing_data[-2]}")
        logging.warning(f"Batch 2 times: {batch_timing_data[-1]}")
        for batch_size in filter((2).__lt__, self.batch_benchmark_sizes):
            # Warm up the model with a batch of the input size count
            new_benchmark_points = calculate_next_benchmark_points(
                batch_timing_data[-2][0],
                batch_timing_data[-2][1],
                self.batch_benchmark_sizes[len(batch_timing_data) - 2],
                batch_timing_data[-1][0],
                batch_timing_data[-1][1],
                self.batch_benchmark_sizes[len(batch_timing_data) - 1],
                points_ratio,
            )
            new_benchmarking_model_inputs = list(map(partial(self.generate_test_input,batch_size), new_benchmark_points))
            batch_timing_data.append((new_benchmark_points, list(reversed(self.batch_inference_times(reversed(new_benchmarking_model_inputs))))))
            logging.warning(f"Batch {batch_size} times: {batch_timing_data[-1]}")
        
        # Pad the benchmark data to have the same number of points for all batch sizes
        for i in range(2, len(batch_timing_data)):
            model_input_padding, timing_padding = get_benchmark_data_paddings(batch_timing_data[i - 1][0], batch_timing_data[i - 1][1], batch_timing_data[i][0], batch_timing_data[i][1], self.batch_benchmark_sizes[i]/self.batch_benchmark_sizes[i-1])
            batch_timing_data[i] = (batch_timing_data[i][0] + model_input_padding, batch_timing_data[i][1] + timing_padding)

        logging.warning(f"Final batch timing data: {batch_timing_data}")
        return batch_timing_data


    def __init__(self, model_pool: ThreadPoolExecutor) -> None:
        """Initialize model/tokenizer state and batching metadata."""
        from src.razors_edge.optimal_batching import get_batch_start_end_idx_and_duration
        from src.razors_edge.optimal_benchmarking import create_batch_timing_estimators

        self.get_batch_start_end_idx_and_duration = get_batch_start_end_idx_and_duration
        self.model = self.load_model(model_pool)
        if self.enable_cache_batch_timing_estimators and self._find_load_model_provider() in RazorsEdgeComputeTask._cache_for_batch_timing_estimators:
            batch_timing_estimators = RazorsEdgeComputeTask._cache_for_batch_timing_estimators[self._find_load_model_provider()]
            self.batch_timing_estimators = batch_timing_estimators
        else:
            batch_timing_data = self.get_batch_timing_data()
            self.batch_timing_estimators = create_batch_timing_estimators(self.batch_benchmark_sizes, batch_timing_data, self.max_input_size, 0)
            RazorsEdgeComputeTask._cache_for_batch_timing_estimators[self._find_load_model_provider()] = self.batch_timing_estimators
        self.expected_schedule_time = time.perf_counter_ns()

    @override
    def preprocess_input_without_size(self, input_data: Any) -> Any:
        """Change input data without adding size if needed"""
        return input_data

    def preprocess_input(self, input_data: Any) -> tuple[Any, int]:
        """Convert external input into a queueable payload with size."""
        preprocessed_input = self.preprocess_input_without_size(input_data)
        return preprocessed_input, self.get_input_size(input_data, preprocessed_input)

    @override
    def create_batch(self, to_batch: list[Any]) -> tuple[tuple, dict]:
        """Create an inference batch from queued items."""

    def get_batch_ids_list_and_batch(self, ids_and_inputs_queue: dict[tuple[int, int], tuple[Any, int]]) -> tuple[Sequence[tuple[int, int]], Any]:
        """Pick ids for the next batch and build batch input."""
        # We unpack the dictionary to obtain the sorted token counts, operation ids and queuing times
        model_input_sizes, operation_ids, queuing_times = zip(
            *sorted((input_size, operation_id, queuing_time) for (operation_id, queuing_time), (_input_data, input_size) in ids_and_inputs_queue.items()),
            strict=True,
        )
        # If the expected schedule time is in the past, set it to the current time. Setting it to current time is slightly inaccurate as this function will take time to execute.
        # At full load, the expected schedule time will be in the future, so this is not a problem.
        self.expected_schedule_time = max(self.expected_schedule_time, time.perf_counter_ns())
        batch_start_idx, batch_end_idx, batch_process_duration = self.get_batch_start_end_idx_and_duration(model_input_sizes, self.batch_timing_estimators, queuing_times, self.expected_schedule_time, self.latency_strategy)
        self.expected_schedule_time += batch_process_duration  # This will be used for the next batch
        # Reconstruct the keys of the queue dictionary for the current batch
        queue_dict_keys: tuple[tuple[int, int], ...] = tuple(zip(operation_ids[batch_start_idx:batch_end_idx], queuing_times[batch_start_idx:batch_end_idx], strict=True))
        to_batch = [ids_and_inputs_queue[i][0] for i in queue_dict_keys]
        # Create the batch
        return queue_dict_keys, self.create_batch(to_batch)

    def __call__(self, batched_inputs: tuple[tuple, dict]) -> Any:
        """Run inference for a prepared batch."""
        return self.model(*batched_inputs[0], **batched_inputs[1])

    @override
    def postprocess_output(self, call_output: Any) -> Iterable[Any]:
        """Map model output to final output."""
