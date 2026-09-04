import multiprocessing
import os
import tempfile
import unittest

import numpy as np

from DEEM.DEEMI import DEEM
from DEEM.evaluation import create_evaluation_executor, evaluate_cost_function
from DEEM.population import CandidateSolution


def sphere(x):
    """Pickleable objective used by spawn-mode process tests."""
    return float(np.dot(x, x))


class ParallelRuntimeTests(unittest.TestCase):
    def test_reusable_process_executor(self):
        candidates = [
            CandidateSolution(np.array([value, 2.0]), sphere)
            for value in (1.0, 3.0, 5.0)
        ]
        executor = create_evaluation_executor(2)
        try:
            _, first_count = evaluate_cost_function(
                candidates, nworkers=2, executor=executor)
            for candidate in candidates:
                candidate.x = candidate.x + 1.0
            _, second_count = evaluate_cost_function(
                candidates, nworkers=2, executor=executor)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        self.assertEqual(first_count, 3)
        self.assertEqual(second_count, 3)
        self.assertEqual(
            [candidate.f for candidate in candidates],
            [13.0, 25.0, 45.0],
        )

    def test_deemi_reports_timing_and_closes_executor(self):
        original_directory = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                optimizer = DEEM(
                    function=sphere,
                    lower_bound=np.array([-2.0, -2.0]),
                    upper_bound=np.array([2.0, 2.0]),
                    nparticles_max=6,
                    nparticles_min=6,
                    npop_max=1,
                    npop_min=1,
                    maxiter=0,
                    nworkers=2,
                    method_subswarm_reduction='constant',
                    method_subswarm_creation='fitness-focused',
                    seed=1234,
                )
                optimizer.log.plot_results = lambda: None
                result = optimizer.update()
            finally:
                os.chdir(original_directory)

        self.assertGreaterEqual(result['initial_evaluation_time'], 0.0)
        self.assertGreaterEqual(result['evaluation_time'], 0.0)
        self.assertGreaterEqual(result['optimizer_overhead_time'], 0.0)
        self.assertIsNone(optimizer._executor)


if __name__ == '__main__':
    if os.environ.get('DEEM_TEST_SPAWN') == '1':
        multiprocessing.set_start_method('spawn', force=True)
    unittest.main()
