import errno
import logging
import multiprocessing as mp
import queue
import signal
import traceback
from typing import Any, Callable, Iterable, List

from ._joboutput import JobOutput

logger = logging.getLogger(__name__)


class MpRunner(object):
    """
    A general purpose multiprocessing runner class. It can be used
    for purely independent jobs without shared writable objects.

    Note: This class is not thread safe when multiple workers are writing the same objects.
    """

    def __init__(
        self,
        num_workers: int,
    ):
        self.num_workers = self._parse_num_workers(num_workers)

    @staticmethod
    def _parse_num_workers(num_workers: int) -> int:
        """parse the number of workers

        num_workers :param: number of workers intended to use
        :return: parsed number of workers to use,
        """
        if 0 <= num_workers <= 1:
            return 0
        elif num_workers < 0:
            return mp.cpu_count() - 1
        else:
            return int(num_workers)

    def _parallel_exec(self, jobs: Iterable, exec_func: Callable) -> List[JobOutput]:
        """handle multiple defined jobs via multiprocessing

        jobs :param: An iterable list that defines the jobs
        :return: None
        """
        logger.info(f"Start running the jobs on {self.num_workers} CPUs.")

        manager = mp.Manager()
        job_queue = manager.Queue()
        for i, job in enumerate(jobs):
            job_queue.put((i, job))
        result_queue = manager.Queue()

        workers = []
        for _ in range(self.num_workers):
            proc = mp.Process(
                target=self._executor_hook, args=(job_queue, result_queue, exec_func)
            )
            proc.start()
            workers.append(proc)

        # synchronize the forked workers and handle keyboard interrupt in main process
        try:
            for worker in workers:
                worker.join()
        except KeyboardInterrupt:
            for worker in workers:
                worker.terminate()
                worker.join()

        _results = []
        while not result_queue.empty():
            _results.append(result_queue.get(block=False))

        return _results

    @staticmethod
    def executor(jobid: int, job: Any, exec_func: Callable) -> JobOutput:
        """job handler"""
        # noinspection PyBroadException
        try:
            exec_rc = exec_func(job)
            return JobOutput(
                jobid=jobid, run_ok=True, result=dict(output=exec_rc, msg="")
            )
        except Exception:
            msg = traceback.format_exc()
            return JobOutput(
                jobid=jobid, run_ok=False, result=dict(output=None, msg=msg)
            )

    def run(self, jobs: Iterable, exec_func: Callable) -> List[JobOutput]:
        """the main func to run the jobs"""
        # global multiprocessing_runner
        # multiprocessing_runner = self
        results = None

        if self.num_workers > 1:
            try:
                results = self._parallel_exec(jobs, exec_func)
            except IOError as ie:
                if ie.errno == errno.EPIPE:
                    raise Exception("Broken pipe error, interrupted.")
        else:
            # sequentially run the job when num_workers <= 1
            results = [self.executor(i, job, exec_func) for i, job in enumerate(jobs)]

        return results

    def _executor_hook(
        self, job_queue: mp.Queue, result_queue: mp.Queue, exec_func: Callable
    ):
        # ignore the CTRL+C signaling
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        while not job_queue.empty():
            # noinspection PyBroadException
            try:
                jobid, job = job_queue.get(block=False)
                result_queue.put(self.executor(jobid, job, exec_func))
            except queue.Empty:
                pass
            except Exception:
                traceback.print_exc()
