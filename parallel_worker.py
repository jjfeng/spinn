import sys
import os
import traceback
import numpy as np
import tensorflow as tf

class ParallelWorker:
    """
    Stores the information for running something in parallel
    These workers can be run throught the ParallelWorkerManager
    """
    def __init__(self, seed):
        """
        @param seed: a seed for for each parallel worker
        """
        raise NotImplementedError()

    def run(self):
        """
        Do not implement this function!
        """
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        result = None
        try:
            result = self.run_worker()
        except Exception as e:
            print("Exception caught in parallel worker: %s", e)
            traceback.print_exc()
        return result

    def run_worker(self):
        """
        Implement this function!
        Returns whatever value needed from this task
        """
        raise NotImplementedError()

    def  __str__(self):
        """
        @return: string for identifying this worker in an error
        """
        raise NotImplementedError()

class ParallelWorkerManager:
    """
    Runs many ParallelWorkers
    """
    def run(self):
        raise NotImplementedError()

class MultiprocessingManager(ParallelWorkerManager):
    """
    Handles submitting jobs to a multiprocessing pool
    We have written our own custom function for batching jobs together
    So runs ParallelWorkers using multiple CPUs on the same machine
    """
    def __init__(self, pool, worker_list):
        """
        @param worker_list: List of ParallelWorkers
        """
        self.pool = pool
        self.worker_list = worker_list

    def run(self):
        try:
            results_raw = self.pool.map(run_multiprocessing_worker, self.worker_list)
        except Exception as e:
            print("Error occured when trying to process workers in parallel: %s", e)
            # Just do it all one at a time instead
            results_raw = map(run_multiprocessing_worker, self.worker_list)

        results = []
        for i, r in enumerate(results_raw):
            if r is None:
                print("WARNING: multiprocessing worker for this worker failed: %s", self.worker_list[i])
            else:
                results.append(r)
        return results

def run_multiprocessing_worker(worker):
    """
    @param worker: ParallelWorker
    Function called on each worker process, used by MultiprocessingManager
    Note: this must be a global function
    """
    return worker.run()
