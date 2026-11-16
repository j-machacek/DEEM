#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Jan Machaček
#
# This file is part of DEEM, released under the BSD 3-Clause License.
# See the LICENSE file in the project root for the full license text.

"""
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#               DEEM - Differential Evolution with Elitism and Multi-populations
#                            Copyright (C) 2023-2026 Jan Machacek
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#
# author: Jan Machacek, jan-machacek@outlook.com
# date: 16.11.2026
#
# Robustness tests of the DEEM core: non-finite objective values, failed
# evaluations, the start value F0, the iteration and evaluation budget, bound
# validation, the optimiser's own random number generator, QMC sampling,
# boundary handling, scale invariance, worker processes, the evaluation cache
# and the surrogate managers.
#
# numgeo-ACT carries the same tests for its vendored copy of the package (only
# the import block differs).
#
# History
# 16.11.2026, J. Machacek - Initial version
#
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
"""

import contextlib
import io
import os
import tempfile
import unittest
import warnings

import numpy as np

# ---- import block (the only part that differs in numgeo-ACT) -------------- #
from DEEM import DEEMI as deemi_module
from DEEM import evaluation as evaluation_module
from DEEM.DEEMI import DEEM as DEEMI, FiniteObjective, validate_bounds, make_rng
from DEEM.boundary_conditions import enforce_BC
from DEEM.evaluation import EvalCache, register_worker_state, get_worker_data
from DEEM.population import Population, CandidateSolution
from DEEM.sampling import sampling, scrambled_qmc_engine
from DEEM.surrogate import SurrogateManager, GPSurrogateManager
from DEEM.toolbox import space_filling_sample, covariance_seed
try:
    from DEEM.DEEM import DEEM as PublishedDEEM
except ImportError:          # the published variant is not part of every copy
    PublishedDEEM = None
# --------------------------------------------------------------------------- #

PENALTY = 1000.0


def sphere(x):
    """Shifted sphere (pickleable for the process-pool tests)."""
    return float(np.sum((np.asarray(x, dtype=float) - 0.3) ** 2))


def nan_region(x):
    return float('nan') if x[0] > 0.5 else sphere(x)


def always_penalty(x):
    return PENALTY


def above_penalty(x):
    # values above the penalty are failures as well
    return 10.0 * PENALTY if x[0] > 0.0 else sphere(x)


def noisy_global_rng(x):
    np.random.rand()          # an objective that draws from numpy's global RNG
    return sphere(x)


_HOOK_STATE = {'offset': 0.0}


def hook_capture():
    return dict(_HOOK_STATE)


def hook_restore(state):
    _HOOK_STATE.update(state)


def offset_objective(x):
    return _HOOK_STATE['offset']


@contextlib.contextmanager
def quiet_directory():
    """Run in a temporary directory (the logger writes to ./) without output."""
    original = os.getcwd()
    with tempfile.TemporaryDirectory() as directory:
        os.chdir(directory)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                yield
        finally:
            os.chdir(original)


def run_deemi(function, lower, upper, **options):
    settings = dict(nparticles_max=12, nparticles_min=12, npop_max=3, npop_min=3,
                    maxiter=6, nworkers=1, penalty=PENALTY)
    settings.update(options)
    with quiet_directory():
        optimizer = DEEMI(function, lower, upper, **settings)
        optimizer.log.plot_results = lambda: None
        optimizer.update()
    return optimizer


def run_published(function, lower, upper, **options):
    settings = dict(nparticles_max=12, nparticles_min=12, nswarm_max=3, nswarm_min=3,
                    maxiter=6, nworkers=1, penalty=PENALTY)
    settings.update(options)
    with quiet_directory():
        optimizer = PublishedDEEM(function, lower, upper, **settings)
        optimizer.log.plot_results = lambda: None
        optimizer.update()
    return optimizer


def trajectory(optimizer):
    return (optimizer.FBEST, optimizer.XBEST.tolist(), optimizer.fev, optimizer.n_restarts,
            [list(map(float, c.x)) for c in optimizer.candidates])


class TestNonFiniteAndFailedEvaluations(unittest.TestCase):
    def test_finite_objective_wrapper(self):
        wrapped = FiniteObjective(lambda x: x[0], PENALTY)
        for bad in (np.nan, np.inf, -np.inf):
            self.assertEqual(wrapped([bad]), PENALTY)
        self.assertEqual(wrapped([0.25]), 0.25)
        self.assertEqual(FiniteObjective(lambda x: None, PENALTY)([0.0]), PENALTY)

    def test_nan_region_never_becomes_the_best(self):
        for seed in range(8):
            optimizer = run_deemi(nan_region, [-2.0, -2.0], [2.0, 2.0], seed=seed)
            self.assertTrue(np.isfinite(optimizer.FBEST))
            self.assertLess(optimizer.FBEST, PENALTY)
            self.assertTrue(all(np.isfinite(f) for f in optimizer.FBEST_history))

    def test_start_vector_in_the_nan_region_is_penalised(self):
        optimizer = run_deemi(nan_region, [-2.0, -2.0], [2.0, 2.0], seed=4, X0=[1.5, 1.5])
        self.assertLess(optimizer.FBEST, PENALTY)

    def test_all_failed_generation_completes(self):
        optimizer = run_deemi(always_penalty, [-1.0, -1.0], [1.0, 1.0], seed=1)
        self.assertEqual(optimizer.FBEST, PENALTY)
        self.assertEqual(optimizer.iters, 6)

    def test_values_above_the_penalty_are_failures(self):
        optimizer = run_deemi(above_penalty, [-1.0, -1.0], [1.0, 1.0], seed=3)
        self.assertTrue(all(p.f < PENALTY for p in optimizer.archive_elite))
        self.assertTrue(optimizer.failed(PENALTY))
        self.assertTrue(optimizer.failed(10.0 * PENALTY))
        self.assertFalse(optimizer.failed(PENALTY * (1.0 - 1e-12)))

    @unittest.skipIf(PublishedDEEM is None, 'published variant not available')
    def test_published_variant_handles_nan_and_failure_plateaus(self):
        np.random.seed(2)
        optimizer = run_published(nan_region, [-2.0, -2.0], [2.0, 2.0])
        self.assertTrue(np.isfinite(optimizer.FBEST))
        self.assertLess(optimizer.FBEST, PENALTY)
        np.random.seed(2)
        optimizer = run_published(always_penalty, [-1.0, -1.0], [1.0, 1.0])
        self.assertEqual(optimizer.FBEST, PENALTY)


class TestBudget(unittest.TestCase):
    def test_maxiter_counts_generations_after_the_initial_population(self):
        optimizer = run_deemi(sphere, [-1.0] * 3, [1.0] * 3, seed=0, maxiter=4)
        self.assertEqual(optimizer.iters, 4)
        self.assertEqual(optimizer.result['nit'], 4)
        self.assertEqual(optimizer.fev, 12 * 5)

    def test_maxfev_stops_after_the_generation_that_reaches_it(self):
        optimizer = run_deemi(sphere, [-1.0] * 3, [1.0] * 3, seed=0, maxiter=50, maxfev=40)
        self.assertEqual(optimizer.fev, 48)

    @unittest.skipIf(PublishedDEEM is None, 'published variant not available')
    def test_published_variant_counts_real_evaluations(self):
        np.random.seed(0)
        optimizer = run_published(sphere, [-1.0] * 3, [1.0] * 3, maxiter=4)
        self.assertEqual(optimizer.iters, 4)
        self.assertEqual(optimizer.fev, 12 * 5)

    def test_restart_tail_blocks_restarts_in_the_final_fraction(self):
        options = dict(seed=3, maxiter=20, niter_reset_global=2, tolerance=1.0e3)
        free = run_deemi(sphere, [-2.0] * 3, [2.0] * 3, restart_tail=0.0, **options)
        default = run_deemi(sphere, [-2.0] * 3, [2.0] * 3, **options)
        self.assertEqual(default.restart_tail, 0.1)
        self.assertLessEqual(default.n_restarts, free.n_restarts)
        never = run_deemi(sphere, [-2.0] * 3, [2.0] * 3, restart_tail=0.99, **options)
        tail = run_deemi(sphere, [-2.0] * 3, [2.0] * 3, restart_tail=0.5, **options)
        self.assertGreater(free.n_restarts, 0)
        self.assertEqual(never.n_restarts, 0)
        self.assertLess(tail.n_restarts, free.n_restarts)
        self.assertLessEqual(tail.global_reset_iter, 10)
        for bad in (-0.1, 1.0):
            with self.assertRaises(ValueError):
                run_deemi(sphere, [-2.0] * 3, [2.0] * 3, restart_tail=bad, **options)

    def test_reset_interval_is_at_least_one_generation(self):
        optimizer = run_deemi(sphere, [-1.0] * 2, [1.0] * 2, seed=0, maxiter=5)
        self.assertEqual(optimizer.niter_reset_global, 1)

    def test_known_start_value_is_not_evaluated_again(self):
        calls = []

        def objective(x):
            calls.append(np.asarray(x, dtype=float).copy())
            return sphere(x)

        optimizer = run_deemi(objective, [-2.0, -2.0], [2.0, 2.0], seed=2, maxiter=1,
                              nparticles_max=6, nparticles_min=6, npop_max=2, npop_min=2,
                              X0=[1.0, 1.0], F0=sphere([1.0, 1.0]))
        self.assertEqual(optimizer.fev, 12)
        self.assertEqual(len(calls), 11)
        self.assertFalse(any(np.array_equal(c, [1.0, 1.0]) for c in calls[:5]))


class TestBoundValidation(unittest.TestCase):
    def test_invalid_bounds_and_start_vectors_are_rejected(self):
        cases = [
            (([1.0], [0.0], None), 'lower_bound > upper_bound'),
            (([0.0, np.nan], [1.0, 1.0], None), 'finite'),
            (([0.0], [1.0, 2.0], None), 'equal length'),
            (([], [], None), 'non-empty'),
            (([0.0, 0.0], [1.0, 1.0], [0.5]), 'length of the bounds'),
            (([0.0, 0.0], [1.0, 1.0], [0.5, 1.5]), 'outside the bounds'),
            (([0.0, 0.0], [1.0, 1.0], [0.5, np.inf]), 'finite'),
        ]
        for (lower, upper, x0), message in cases:
            with self.subTest(lower=lower, upper=upper, x0=x0):
                with self.assertRaisesRegex(ValueError, message):
                    validate_bounds(lower, upper, x0)
        with self.assertRaises(ValueError):
            run_deemi(sphere, [1.0, 0.0], [0.0, 1.0], seed=0)

    def test_fixed_parameter_is_allowed(self):
        optimizer = run_deemi(sphere, [-1.0, 0.25], [1.0, 0.25], seed=5)
        self.assertTrue(np.all(np.array([c.x[1] for c in optimizer.candidates]) == 0.25))
        self.assertEqual(optimizer.XBEST[1], 0.25)

    def test_boundary_handling_with_equal_bounds(self):
        for method in ('periodic', 'damping-periodic', 'damping', 'random', 'clip',
                       'damping-periodic-random', 'damping-periodic-clip'):
            with self.subTest(method=method):
                result = enforce_BC([3.0, 0.5], [1.0, 0.0], [1.0, 1.0], [1.0, 0.5], method)
                self.assertEqual(result, [1.0, 0.5])

    def test_boundary_handling_keeps_points_inside(self):
        rng = np.random.RandomState(0)
        for method in ('periodic', 'damping-periodic', 'damping', 'random', 'clip',
                       'damping-periodic-random', 'damping-periodic-clip'):
            for _ in range(200):
                x = rng.uniform(-30.0, 30.0, 3)
                y = enforce_BC(x, [-1.0, 0.0, 5.0], [1.0, 2.0, 6.0], [0.0, 1.0, 5.5], method, rng=rng)
                self.assertTrue(np.all(np.array(y) >= [-1.0, 0.0, 5.0]) and np.all(np.array(y) <= [1.0, 2.0, 6.0]))


class TestRandomNumberGenerator(unittest.TestCase):
    def test_seeded_runs_are_reproducible_with_restarts_and_halton(self):
        options = dict(seed=21, sampling_method='Halton', maxiter=12, niter_reset_global=2,
                       tolerance=1.0e3)
        first = run_deemi(sphere, [-2.0] * 3, [2.0] * 3, **options)
        second = run_deemi(sphere, [-2.0] * 3, [2.0] * 3, **options)
        self.assertGreater(first.n_restarts, 0)
        self.assertEqual(trajectory(first), trajectory(second))

    def test_objective_drawing_random_numbers_does_not_change_the_run(self):
        options = dict(seed=8, maxiter=10, niter_reset_global=2, tolerance=1.0e3)
        clean = run_deemi(sphere, [-2.0] * 3, [2.0] * 3, **options)
        noisy = run_deemi(noisy_global_rng, [-2.0] * 3, [2.0] * 3, **options)
        self.assertEqual(trajectory(clean), trajectory(noisy))

    def test_global_rng_is_not_touched(self):
        np.random.seed(123)
        expected = np.random.rand(3)
        np.random.seed(123)
        run_deemi(sphere, [-2.0] * 2, [2.0] * 2, seed=4)
        np.testing.assert_array_equal(np.random.rand(3), expected)

    def test_integer_seed_reproduces_the_former_global_seeding(self):
        options = dict(maxiter=8, niter_reset_global=2, tolerance=1.0e3)
        seeded = run_deemi(sphere, [-2.0] * 3, [2.0] * 3, seed=11, **options)
        np.random.seed(11)
        legacy = run_deemi(sphere, [-2.0] * 3, [2.0] * 3, seed=None, **options)
        self.assertEqual(trajectory(seeded), trajectory(legacy))

    def test_seed_types(self):
        self.assertIs(make_rng(None), np.random.mtrand._rand)
        state = np.random.RandomState(3)
        self.assertIs(make_rng(state), state)
        a = make_rng(np.random.default_rng(5)).rand(4)
        b = make_rng(np.random.default_rng(5)).rand(4)
        np.testing.assert_array_equal(a, b)
        for bad in (1.5, 'seed', True):
            with self.assertRaises(TypeError):
                make_rng(bad)

    def test_qmc_engines_follow_the_given_rng(self):
        first = (space_filling_sample(8, [0.0, 0.0], [1.0, 1.0], rng=np.random.RandomState(5)),
                 sampling(8, 2, [0.0, 0.0], [1.0, 1.0], 'Halton', rng=np.random.RandomState(5)))
        second = (space_filling_sample(8, [0.0, 0.0], [1.0, 1.0], rng=np.random.RandomState(5)),
                  sampling(8, 2, [0.0, 0.0], [1.0, 1.0], 'Halton', rng=np.random.RandomState(5)))
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        np.random.seed(None)
        self.assertFalse(np.array_equal(space_filling_sample(8, [0.0, 0.0], [1.0, 1.0]),
                                        space_filling_sample(8, [0.0, 0.0], [1.0, 1.0])))
        self.assertIsNotNone(scrambled_qmc_engine.__doc__)

    def test_results_do_not_depend_on_the_number_of_workers(self):
        options = dict(seed=6, maxiter=5, niter_reset_global=2, tolerance=1.0e3)
        serial = run_deemi(sphere, [-2.0] * 3, [2.0] * 3, nworkers=1, **options)
        parallel = run_deemi(sphere, [-2.0] * 3, [2.0] * 3, nworkers=2, **options)
        self.assertEqual(trajectory(serial), trajectory(parallel))


class TestSampling(unittest.TestCase):
    def test_sobol_without_external_package(self):
        a = sampling(10, 3, [0.0] * 3, [1.0] * 3, 'Sobol')
        b = sampling(10, 3, [0.0] * 3, [1.0] * 3, 'Sobol')
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (10, 3))
        self.assertFalse(np.any(np.all(a == 0.0, axis=1)))      # the all-zero point is skipped

    def test_lhs_is_a_latin_hypercube(self):
        n, d = 16, 4
        samples = sampling(n, d, [0.0] * d, [1.0] * d, 'LHS', rng=np.random.RandomState(1))
        for j in range(d):
            np.testing.assert_array_equal(np.sort(np.floor(samples[:, j] * n)), np.arange(n))
        # the dimensions are not all equally ordered (no diagonal)
        orders = {tuple(np.argsort(samples[:, j])) for j in range(d)}
        self.assertGreater(len(orders), 1)

    def test_lhs_run_with_restarts_emits_no_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            optimizer = run_deemi(sphere, [-2.0] * 3, [2.0] * 3, seed=3, maxiter=12,
                                  niter_reset_global=2, tolerance=1.0e3, sampling_method='LHS')
        self.assertGreater(optimizer.n_restarts, 0)
        self.assertEqual([str(w.message) for w in caught], [])


class TestScaleInvariance(unittest.TestCase):
    """A per-dimension power-of-two rescaling of the parameters is exact in
    floating point, so a scale-invariant optimiser follows the identical path."""

    FACTORS = np.array([2.0 ** -10, 1.0, 2.0 ** 12])

    @staticmethod
    def scaled(function, factors):
        return lambda y: function(np.asarray(y) / factors)

    def check(self, **options):
        lower, upper = np.array([-2.0, -1.0, -3.0]), np.array([2.0, 3.0, 1.0])
        base = run_deemi(sphere, lower, upper, **options)
        f = self.FACTORS
        scaled = run_deemi(self.scaled(sphere, f), lower * f, upper * f, **options)
        self.assertEqual(base.FBEST, scaled.FBEST)
        np.testing.assert_array_equal(base.XBEST * f, scaled.XBEST)
        self.assertEqual(base.n_restarts, scaled.n_restarts)
        self.assertEqual(base.DIV_NORM_GB, scaled.DIV_NORM_GB)

    def test_equally_distributed_sub_populations_without_restart(self):
        self.check(seed=1, maxiter=15, restart_budget=0, npop_max=4, npop_min=2)

    def test_fitness_focused_with_density_restarts(self):
        self.check(seed=2, maxiter=15, npop_max=4, npop_min=2, method_reset='density',
                   method_subswarm_creation='fitness-focused', method_subswarm_reduction='exponential',
                   method_boundary='damping-periodic', niter_reset_global=2, tolerance=1.0e3)

    def test_scaled_positions(self):
        X = np.array([[1.0, 10.0, 0.0], [3.0, 30.0, 0.0]])
        Y = Population.scaled_positions(X, [0.0, 0.0, 0.0], [2.0, 20.0, 0.0])
        np.testing.assert_allclose(Y, [[10.0, 10.0, 0.0], [30.0, 30.0, 0.0]])
        same = Population.scaled_positions(X, [0.0] * 3, [5.0] * 3)
        np.testing.assert_array_equal(same, X)

    def test_covariance_seed_respects_small_ranges(self):
        rng = np.random.RandomState(0)
        lower, upper = np.array([0.0, 0.0]), np.array([1e6, 1e-3])
        members = np.column_stack([rng.uniform(0, 1e6, 20), rng.uniform(0, 1e-3, 20)])
        samples = covariance_seed(np.array([5e5, 5e-4]), members, lower, upper, n=200, rng=rng)
        # the small dimension is not swamped by a ridge of the size of the large one
        self.assertLess(np.std(samples[:, 1]), 1e-3)
        self.assertGreater(len(np.unique(samples[:, 1])), 100)


class TestWorkers(unittest.TestCase):
    def setUp(self):
        # a host application (numgeo-ACT) may have registered its own hook
        self._previous_hook = evaluation_module._worker_state

    def tearDown(self):
        evaluation_module._worker_state = self._previous_hook

    def test_without_hook_there_is_no_worker_data(self):
        register_worker_state(None, None)
        self.assertIsNone(get_worker_data())

    def test_hook_validation(self):
        with self.assertRaises(TypeError):
            register_worker_state(hook_capture, 'not callable')

    def test_registered_state_reaches_the_worker_processes(self):
        _HOOK_STATE['offset'] = 7.0
        register_worker_state(hook_capture, hook_restore)
        try:
            executor = evaluation_module.create_evaluation_executor(2)
            candidates = [CandidateSolution(np.array([float(i)]), offset_objective) for i in range(3)]
            try:
                evaluation_module.evaluate_cost_function(candidates, nworkers=2, executor=executor)
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
        finally:
            _HOOK_STATE['offset'] = 0.0
        self.assertEqual([c.f for c in candidates], [7.0, 7.0, 7.0])


class TestCacheAndSurrogate(unittest.TestCase):
    def test_cache_answers_repeated_positions(self):
        cache = EvalCache([0.0, 0.0], [1.0, 1.0], rel_tol=1e-9)
        calls = []

        def objective(x):
            calls.append(1)
            return sphere(x)

        candidates = [CandidateSolution(np.array([0.2, 0.4]), objective) for _ in range(3)]
        _, n_real = evaluation_module.evaluate_cost_function(candidates[:1], cache=cache)
        _, n_cached = evaluation_module.evaluate_cost_function(candidates[1:], cache=cache)
        self.assertEqual((n_real, n_cached, len(calls)), (1, 0, 1))
        self.assertEqual(candidates[2].f, candidates[0].f)

    def test_cached_run_never_counts_more_than_the_uncached_one(self):
        plain = run_deemi(sphere, [-1.0] * 2, [1.0] * 2, seed=9, maxiter=8)
        cached = run_deemi(sphere, [-1.0] * 2, [1.0] * 2, seed=9, maxiter=8, cache_tol=1e-12)
        self.assertLessEqual(cached.fev, plain.fev)
        self.assertEqual(cached.result['cache_size'] > 0, True)

    def test_surrogate_managers_select_valid_indices(self):
        for manager_class in (SurrogateManager, GPSurrogateManager):
            with self.subTest(manager=manager_class.__name__):
                rng = np.random.RandomState(0)
                manager = manager_class([-1.0, -1.0], [1.0, 1.0], eval_frac=0.5, explore_frac=0.3,
                                        min_train=10, penalty=PENALTY)
                warm = [CandidateSolution(rng.uniform(-1, 1, 2), sphere) for _ in range(12)]
                self.assertEqual(manager.select(None, warm), list(range(12)))
                evaluation_module.evaluate_cost_function(warm)
                manager.observe(warm)
                trial = [CandidateSolution(rng.uniform(-1, 1, 2), sphere) for _ in range(12)]
                for c in trial:
                    c.initialized = True
                chosen = manager.select(None, trial)
                self.assertEqual(chosen, sorted(set(chosen)))
                self.assertTrue(all(0 <= i < 12 for i in chosen))
                self.assertEqual(len(chosen), 6)

    def test_deemi_with_surrogate_saves_evaluations_and_keeps_pairs_consistent(self):
        manager = SurrogateManager([-2.0] * 2, [2.0] * 2, eval_frac=0.5, explore_frac=0.3,
                                   min_train=12, penalty=PENALTY)
        optimizer = run_deemi(sphere, [-2.0] * 2, [2.0] * 2, seed=4, maxiter=10, surrogate=manager)
        self.assertLess(optimizer.fev, 12 * 11)
        self.assertGreater(manager.n_saved, 0)
        for c in optimizer.candidates:
            if not c.improved:
                self.assertTrue(c.f >= c.fbest)


class TestDiversity(unittest.TestCase):
    def test_diversity_does_not_depend_on_the_parameter_units(self):
        optimizer = run_deemi(sphere, [-1.0, -1.0], [1.0, 1.0], seed=0, maxiter=1)
        rng = np.random.RandomState(1)
        X0, X1 = rng.uniform(-1, 1, (10, 2)), rng.uniform(-0.1, 0.1, (10, 2))
        reference = optimizer.diversity(X1) / optimizer.diversity(X0)
        optimizer.LB = np.array([-1.0, -1000.0]); optimizer.UB = np.array([1.0, 1000.0])
        scale = np.array([1.0, 1000.0])
        ratio = optimizer.diversity(X1 * scale) / optimizer.diversity(X0 * scale)
        self.assertAlmostEqual(ratio, reference, places=12)
        # in raw units the wide parameter would dominate the measure
        raw = np.mean(np.abs(np.median(X1 * scale, axis=0) - X1 * scale)) /             np.mean(np.abs(np.median(X0 * scale, axis=0) - X0 * scale))
        self.assertNotAlmostEqual(raw, reference, places=3)

    def test_density_restart_without_any_successful_evaluation(self):
        optimizer = run_deemi(always_penalty, [-1.0, -1.0], [1.0, 1.0], seed=1, maxiter=8,
                              method_reset='density', niter_reset_global=2, tolerance=1.0)
        self.assertGreater(optimizer.n_restarts, 0)


if __name__ == '__main__':
    unittest.main()
