"""A file containing the base compute task. Should not have any module level imports."""

from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, override

class BaseBatchedComputeTask:
    """
    Base class for any kind of compute that needs to run.
    Accept in batch is a primitive way to decide if somthing should be added to the batch element by element
    """

    @override
    def __init__(self, model_pool: ThreadPoolExecutor) -> None:
        """Load the model(s) as instance variable(s). Model pool is passed incase we want to use the threadpool for some loading."""
        raise NotImplementedError("NEEDS TO BE IMPLEMENTED")

    def preprocess_input(self, input_data: Any) -> Any:
        """Preprocess the input data if needed. Default is to return the input data as is, can contain operations like tokenization."""
        return input_data

    def _accept_in_batch(self, current_batch: list, candidate: Any) -> bool:
        """Check if a candidate belongs to the current batch based on text size, current size etc. False by default for batch size 1."""
        return False

    def get_batch_ids_list_and_batch(self, ids_and_inputs_queue: dict[tuple[int, int], Any]) -> tuple[Sequence[tuple[int, int]], Any]:
        """Sample Implementation. Create a batch for ComputeBase with index target_id from all_data. Create the ids, batch and picked_indexes."""
        ids_and_inputs_iter = iter(ids_and_inputs_queue.items())
        first_id, first_input = next(ids_and_inputs_iter)
        ids, batch = [first_id], [first_input]
        for id, input_data in ids_and_inputs_iter:
            if self._accept_in_batch(batch, input_data):
                ids.append(id)
                batch.append(input_data)
            else:
                break
        return ids, batch

    @override
    def __call__(self, batched_inputs: Any) -> Any:
        """Run the computing task on a batch of inputs."""
        raise NotImplementedError("NEEDS TO BE IMPLEMENTED")

    def postprocess_output(self, call_output: Any) -> Iterable[Any]:
        """Post process the outputs of the computing task. Default is to return the outputs as is."""
        return call_output
