"""Contains the definition of ComputeExecutor which can take BaseBatchedComputeTasks as a plugin."""

import asyncio
import logging
from asyncio import Future as AsyncFuture
from collections.abc import Sequence
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pipe, connection, get_context, spawn
from threading import Lock, Semaphore, Thread
from time import perf_counter_ns, sleep
from typing import Any, Iterable
import sys

from src.executor.base_batched_compute_task import BaseBatchedComputeTask


class ComputeExecutor:
    """An executor that accepts a list of BaseBatchedComputeTask classes which it initializes and runs in a separate process."""

    def __init__(self, compute_targets: Iterable[type[BaseBatchedComputeTask]], async_limit: int = 2, model_thread_limit: int = 1):
        """Start a process and a thread with the given BaseBatchedComputeTasks and limits."""
        # Synchronization Primitives
        self._compute_targets_dict = {task_class: i for i, task_class in enumerate(compute_targets)}
        self._pending_futures: dict[int, AsyncFuture | ConcurrentFuture] = dict()
        self._operation_id = 2**33  # hitting beyond 32 bit limit on purpose.
        # Asyncio async_compute_fn Primitives
        self._async_limit_semaphore = asyncio.Semaphore(async_limit + 1)
        # Multithreaded compute_fn Primitives
        self._thread_lock = Lock()
        self._thread_semaphore = Semaphore(async_limit + 1)
        # Process Pipes and Thread Pools
        self.send_pool = ThreadPoolExecutor(1)  # only one thread to send results back with 0 backpressure
        # Pipes
        get_input, self._send_input = Pipe(duplex=False)
        get_time, self._send_time = Pipe(duplex=False)
        self._get_output, send_output = Pipe(duplex=False)
        # New Process
        model_process = get_context("spawn").Process(target=self._InternalProcess, args=(get_input, get_time, send_output, tuple(self._compute_targets_dict.keys()), model_thread_limit), daemon=True)
        old_spd = spawn.get_preparation_data
        spawn.get_preparation_data = lambda _: {"sys_path": sys.path.copy()}  # Reduces new process memory
        model_process.start()
        spawn.get_preparation_data = old_spd
        # Process Startup signal
        big_data = b"A" * (10 * 1024**2)  # 10 megabyte of data
        logging.info("SENDING STARTUP SIGNAL PING ASYNCHRONOUSLY")
        self.send_pool.submit(self._send_input.send, big_data)
        while not self._get_output.poll(5.0):
            if model_process.is_alive():
                logging.warning("MODEL PROCESS ALIVE, WAITING FOR STARTUP SIGNAL PING")
            else:
                raise SystemExit("MODEL PROCESS DEAD, SHUTTING DOWN")
        if self._get_output.recv() != big_data:
            raise SystemExit("STARTUP SIGNAL BROKEN WITH LARGE DATA")
        logging.info("STARTUP SIGNAL PING RECEIVED")
        self._send_input.send(perf_counter_ns())  # send the perf_counter_ns offset to the process
        Thread(target=self.set_time_loop, daemon=True).start()  # start the time loop
        logging.info("PERF COUNTER OFFSET SENT")
        # New Thread
        set_result_thread = Thread(target=self.set_result_loop, daemon=True)
        set_result_thread.start()
        self.healthy = lambda: set_result_thread.is_alive() and model_process.is_alive()
        self.__del__ = lambda: send_output.send(None) and model_process.terminate()
        logging.info("THREAD STARTED")

    class _InternalProcess:
        """A class that is executed in a different process."""

        @classmethod
        def __init__(cls, get_input: connection.Connection, get_time: connection.Connection, send_output: connection.Connection, compute_targets: tuple[type[BaseBatchedComputeTask]], model_thread_limit: int):
            """Run the while True loop in a daemon process."""
            # primitives
            from threading import Semaphore

            cls.model_pool = ThreadPoolExecutor(model_thread_limit)  # Main threadpool for running models
            cls.send_pool = ThreadPoolExecutor(1)  # only one thread to send results back
            cls.pool_limiter = Semaphore(model_thread_limit + 1)  # +1 for the main thread that accumulates data
            cls.get_input = get_input
            cls.get_time = get_time
            cls.send_output = send_output
            # load all models by calling __init__ on them
            logging.warning("INITIALIZING ALL COMPUTE TARGETS")
            initialized_compute = [i(cls.model_pool) for i in compute_targets]
            all_queues: list[dict[tuple[int, int], Any]] = [dict() for _ in compute_targets]
            logging.warning("ALL COMPUTE TARGETS INITIALIZED")
            # be polite and let the garbage collector know it doesn't have to clean up models
            import gc

            gc.collect()
            gc.freeze()
            # startup signal indicates models are loaded
            send_output.send(get_input.recv())  # send the initial data back to the main process, if it is not received, the process will exit
            logging.warning("START SIGNAL PING RESPONSE SENT, COMPUTE LOOP STARTED")
            other_process_time = cls.get_input.recv()
            cls.perf_counter_offset = perf_counter_ns() - other_process_time  # offset for perf_counter_ns to get the correct time
            Thread(target=cls.get_time_loop, daemon=True).start()  # start the time loop
            logging.warning(f"PERF COUNTER OFFSET RECEIVED {cls.perf_counter_offset} AND THREAD STARTED")
            # initialize all_data and start the loop
            while True:
                cls.pool_limiter.acquire()
                cls.accumulate_data(all_queues, initialized_compute)
                fair_queue_index = cls.choose_fair_queue(all_queues)
                batch_ids, input_batch = initialized_compute[fair_queue_index].get_batch_ids_list_and_batch(all_queues[fair_queue_index])
                cls.remove_ids_from_queue(all_queues[fair_queue_index], batch_ids)
                result = cls.model_pool.submit(initialized_compute[fair_queue_index], input_batch)  # run the model in a different thread
                cls.send_pool.submit(cls.post_process_and_send_result, initialized_compute[fair_queue_index], result, batch_ids)

        @classmethod
        def accumulate_data(cls, all_queues: list[dict[tuple[int, int], Any]], initialized_compute: list[BaseBatchedComputeTask]):
            """Add some data to all_data, block if all_data is empty."""
            # recv is blocking, but poll indicates if recv has immediate data available, so we put the "not all_data" condition to block until we have some data
            while cls.get_input.poll() or not any(all_queues):
                compute_id, (operation_id, queue_time, input_data) = cls.get_input.recv()
                # Preprocessing creates a delay to accomodate more items, but it also creates backpressure which needs to be removed by the send_pool in the main process
                all_queues[compute_id][(operation_id, queue_time + cls.perf_counter_offset)] = initialized_compute[compute_id].preprocess_input(input_data)

        @staticmethod
        def choose_fair_queue(all_queues: list[dict[tuple[int, int], Any]]) -> int:
            """Choose the queue with the smallest order id in the first element."""
            fair_index = None
            for i, queue in enumerate(all_queues):
                if queue and (fair_index is None or next(iter(queue.keys())) < next(iter(all_queues[fair_index].keys()))):
                    fair_index = i
            if fair_index is None:
                logging.critical("QUEUE IS EMPTY AFTER ACCUMULATION, IMPOSSIBLE CONDITION, SHUTTING DOWN")
                raise SystemExit("QUEUE IS EMPTY AFTER ACCUMULATION, IMPOSSIBLE CONDITION, SHUTTING DOWN")
            return fair_index

        @staticmethod
        def remove_ids_from_queue(queue: dict[tuple[int, int], Any], input_ids: Sequence[tuple[int, int]]):
            """Remove the ids from the queue."""
            for i in input_ids:
                del queue[i]

        @classmethod
        def post_process_and_send_result(cls, compute_target: BaseBatchedComputeTask, result: Any, input_ids: Sequence[tuple[int, int]]):
            """Run the model in a different thread and send the results back."""
            # cls.pool_limiter.release() # Release should be here if limiter is = size of model_thread_limit and post processing is slower than preprocessing
            outputs = compute_target.postprocess_output(result.result())  # postprocess the output after awaiting the future
            for (operation_id, offset_queueing_time), outputs in zip(input_ids, outputs, strict=True):
                cls.send_output.send((operation_id, outputs))
            cls.pool_limiter.release()  # Release should be here if limiter = 1 + size of model_thread_limit

        @classmethod
        def get_time_loop(cls):
            """Continuously get the time from the process manager every 10 seconds to prevent desynchronization."""
            while True:
                other_process_time = cls.get_time.recv()
                cls.perf_counter_offset = perf_counter_ns() - other_process_time
                logging.warning(f"PERF COUNTER OFFSET UPDATED: {cls.perf_counter_offset}")

    def set_result_loop(self):
        """Continuously Set the result for async_compute_fn and release the limit_semaphore. Runs in a separate thread."""
        try:
            while True:
                operation_id, data = self._get_output.recv()
                future = self._pending_futures.pop(operation_id)
                if isinstance(future, AsyncFuture):
                    # Handle asyncio future
                    loop = future.get_loop()
                    loop.call_soon_threadsafe(future.set_result, data)
                    loop.call_soon_threadsafe(self._async_limit_semaphore.release)
                # Check if this is a threading future or asyncio future
                elif isinstance(future, ConcurrentFuture):
                    future.set_result(data)
                    self._thread_semaphore.release()
                else:
                    logging.error(f"Unknown future type: {type(future)}. This should not happen.")
                    raise SystemExit("Unknown future type, shutting down.")
        except TypeError:
            logging.critical("Exiting ComputeExecutor")

    def set_time_loop(self):
        """Continuously Set the time for the process manager every 10 seconds to prevent desynchronization."""
        while True:
            sleep(10)
            self._send_time.send(perf_counter_ns())

    async def async_compute_fn(self, compute_target: type[BaseBatchedComputeTask], input_data):
        """Process the input_data in batches to compute_target in a different process. Result set in set_result_loop."""
        if not self.healthy():
            raise SystemExit("EXECUTOR PROCESS DEAD, SHUTTING DOWN")
        future: AsyncFuture = asyncio.get_running_loop().create_future()
        queueing_time = perf_counter_ns()
        await self._async_limit_semaphore.acquire()
        self._operation_id += 1
        self._pending_futures[self._operation_id] = future
        data_to_send = (self._compute_targets_dict[compute_target], (self._operation_id, queueing_time, input_data))
        self.send_pool.submit(self._send_input.send, data_to_send)
        return await future

    def sync_compute_fn(self, compute_target: type[BaseBatchedComputeTask], input_data):
        """Process the input_data in batches to compute_target in a different process. Thread-safe synchronous version."""
        with self._thread_lock:
            if not self.healthy():
                raise SystemExit("EXECUTOR PROCESS DEAD, SHUTTING DOWN")
            future: ConcurrentFuture = ConcurrentFuture()
            queueing_time = perf_counter_ns()
            self._thread_semaphore.acquire()
            self._operation_id += 1
            self._pending_futures[self._operation_id] = future
            data_to_send = (self._compute_targets_dict[compute_target], (self._operation_id, queueing_time, input_data))
            self.send_pool.submit(self._send_input.send, data_to_send)
        return future.result()
