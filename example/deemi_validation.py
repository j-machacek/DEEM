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
# Validation benchmark of DEEMI against the published DEEM and SciPy's
# differential_evolution at equal evaluation budgets.
#
#   * Test functions: shifted (optimum away from the origin and the centre of
#     the box) and, where marked, rotated (separability cannot be exploited)
#     standard functions, split into a DEVELOPMENT set (used for tuning) and a
#     HOLD-OUT set (used to judge a change). Two special suites test a region in
#     which the objective returns NaN and badly scaled parameters (ranges that
#     differ by six orders of magnitude).
#   * Dimensions 5, 10, 20; budgets 100*d (expensive regime, 500 to 2000
#     evaluations) and 2000*d (large regime); 25 seeds per configuration.
#   * Every algorithm sees the same objective wrapper, which records the best
#     true objective value among the first B evaluations (evaluations beyond the
#     budget B are ignored), so budgets are exactly equal.
#   * Statistics are rank based: median and IQR per configuration,
#     Mann-Whitney U (two-sided, 5 %) win/tie/loss counts per configuration,
#     the Wilcoxon signed-rank test across configurations (paired medians), the
#     mean rank, and success rates to a target precision.
#
# Usage (from the repository root):
#   python example/deemi_validation.py run --suite all --regime all --jobs 32 \
#          --out validation/results.csv
#   python example/deemi_validation.py report validation/results.csv \
#          --reference deemi --out validation
#
# Algorithms are named  [impl:]name[+option=value...], e.g.
#   deemi                  DEEMI with its defaults
#   deemi-act              DEEMI with the settings numgeo-ACT uses
#   deem                   the published DEEM (DEEM/DEEM.py)
#   scipy-de               scipy.optimize.differential_evolution (defaults, no polish)
#   deemi+method_reset='density'      an option switched for an experiment
#   old:deemi              DEEMI from another copy of the package, registered with
#                          --impl old=PARENT_DIRECTORY:PACKAGE_NAME
#   deemi+np_per_dim=2     harness option: population of 2 * d (np_min_frac: final
#                          population as a fraction of the initial one)
# report --labels 'spec=Name;spec=Name' sets display names.
#
# History
# 16.11.2026, J. Machacek - Initial version
#
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
"""

import argparse
import ast
import contextlib
import csv
import importlib
import io
import itertools
import math
import os
import sys
import tempfile
import time
import zlib
from concurrent import futures

import numpy as np

REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIMENSIONS = (5, 10, 20)
REGIMES = {'expensive': 100, 'large': 2000}     # budget = factor * dimension
SEEDS = 25
TARGET = 1e-8                                  # success: error <= TARGET
CHECKPOINTS = (0.1, 0.25, 0.5)                 # best-so-far recorded at these budget fractions


# =========================================================================== #
#  Test problems                                                               #
# =========================================================================== #
def _instance_rng(name, dim):
    """Problem instance (shift, rotation) depends on name and dimension only."""
    return np.random.RandomState(zlib.crc32(f"{name}-{dim}".encode()))


def _rotation(rng, dim):
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)))
    return q * np.sign(np.diag(r))


class Problem:
    """
    Shifted/rotated test function on the box [-5, 5]^d with f* = 0 at x* = o.

    scale (optional) rescales the parameters: the optimiser sees y = scale * x
    on the box scale * [-5, 5]^d (badly scaled parameters).
    """

    def __init__(self, kind, dim, rotate, scale=None, nan_region=False):
        self.kind = kind
        self.dim = dim
        rng = _instance_rng(kind + ('-rot' if rotate else ''), dim)
        self.o = rng.uniform(-4.0, 4.0, dim)
        self.Q = _rotation(rng, dim) if rotate else None
        self.scale = None if scale is None else np.asarray(scale, dtype=float)
        self.normal = None
        if nan_region:
            n = rng.normal(size=dim)
            self.normal = n / np.linalg.norm(n)
        self.lb = -5.0 * np.ones(dim)
        self.ub = 5.0 * np.ones(dim)
        if self.scale is not None:
            self.lb = self.lb * self.scale
            self.ub = self.ub * self.scale

    def __call__(self, y):
        x = np.asarray(y, dtype=float)
        if self.scale is not None:
            x = x / self.scale
        z = x - self.o
        if self.normal is not None and float(self.normal @ z) > 0.5:
            return float('nan')
        if self.Q is not None:
            z = self.Q @ z
        return float(FUNCTIONS[self.kind](z))


def _sphere(z):
    return np.sum(z * z)


def _ellipsoid(z):
    d = z.size
    return np.sum(10.0 ** (6.0 * np.arange(d) / max(d - 1, 1)) * z * z)


def _rosenbrock(z):
    x = z + 1.0
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2)


def _rastrigin(z):
    return 10.0 * z.size + np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z))


def _ackley(z):
    d = z.size
    value = (-20.0 * np.exp(-0.2 * np.sqrt(np.sum(z * z) / d))
             - np.exp(np.sum(np.cos(2.0 * np.pi * z)) / d) + 20.0 + np.e)
    return max(value, 0.0)


def _bent_cigar(z):
    return z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2)


def _schwefel12(z):
    return np.sum(np.cumsum(z) ** 2)


def _griewank(z):
    w = 100.0 * z
    i = np.arange(1, z.size + 1)
    return max(1.0 + np.sum(w * w) / 4000.0 - np.prod(np.cos(w / np.sqrt(i))), 0.0)


def _levy(z):
    w = 1.0 + z / 4.0
    term = (np.sin(np.pi * w[0]) ** 2
            + np.sum((w[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2))
            + (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[-1]) ** 2))
    return max(term, 0.0)


_WK = np.arange(21)


def _weierstrass(z):
    a, b = 0.5, 3.0
    w = 0.1 * z
    ak = a ** _WK
    bk = b ** _WK
    inner = np.sum(ak[None, :] * np.cos(2.0 * np.pi * bk[None, :] * (w[:, None] + 0.5)), axis=1)
    value = np.sum(inner) - z.size * np.sum(ak * np.cos(np.pi * bk))
    return max(value, 0.0)


FUNCTIONS = {
    'sphere': _sphere, 'ellipsoid': _ellipsoid, 'rosenbrock': _rosenbrock,
    'rastrigin': _rastrigin, 'ackley': _ackley, 'bent_cigar': _bent_cigar,
    'schwefel12': _schwefel12, 'griewank': _griewank, 'levy': _levy,
    'weierstrass': _weierstrass,
}

# name: (kind, rotated, options)
SUITES = {
    'development': {
        'sphere': ('sphere', False, {}),
        'ellipsoid-rot': ('ellipsoid', True, {}),
        'rosenbrock': ('rosenbrock', False, {}),
        'rastrigin-rot': ('rastrigin', True, {}),
        'ackley-rot': ('ackley', True, {}),
    },
    'holdout': {
        'bent_cigar-rot': ('bent_cigar', True, {}),
        'schwefel12': ('schwefel12', False, {}),
        'griewank-rot': ('griewank', True, {}),
        'levy-rot': ('levy', True, {}),
        'weierstrass-rot': ('weierstrass', True, {}),
    },
    'special': {
        'ellipsoid-rot-nan': ('ellipsoid', True, {'nan_region': True}),
        'rastrigin-rot-nan': ('rastrigin', True, {'nan_region': True}),
        'ellipsoid-rot-scaled': ('ellipsoid', True, {'badly_scaled': True}),
        'rastrigin-rot-scaled': ('rastrigin', True, {'badly_scaled': True}),
        'rosenbrock-scaled': ('rosenbrock', False, {'badly_scaled': True}),
    },
}


def make_problem(name, dim):
    for suite in SUITES.values():
        if name in suite:
            kind, rotate, options = suite[name]
            scale = None
            if options.get('badly_scaled'):
                # ranges from 1e-3 to 1e3 times the reference range, permuted
                rng = _instance_rng(name + '-scale', dim)
                scale = 10.0 ** rng.permutation(np.linspace(-3.0, 3.0, dim))
            return Problem(kind, dim, rotate, scale=scale, nan_region=options.get('nan_region', False))
    raise KeyError(name)


class BudgetRecorder:
    """Objective wrapper: best true value among the first `budget` evaluations."""

    def __init__(self, problem, budget):
        self.problem = problem
        self.budget = budget
        self.n = 0
        self.best = math.inf
        self.marks = {int(math.ceil(fraction * budget)): fraction for fraction in CHECKPOINTS}
        self.at = {}

    def __call__(self, x):
        value = self.problem(x)
        self.n += 1
        if self.n <= self.budget and value == value and value < self.best:
            self.best = value
        if self.n in self.marks:
            self.at[self.marks[self.n]] = self.best
        return value


# =========================================================================== #
#  Algorithms                                                                  #
# =========================================================================== #
IMPLEMENTATIONS = {'current': (REPOSITORY, 'DEEM')}


def _load(impl, module):
    parent, package = IMPLEMENTATIONS[impl]
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return importlib.import_module(f'{package}.{module}')


def _deemi_defaults(dim, budget):
    npart = 50
    return dict(nparticles_max=npart, nparticles_min=npart)


def _deemi_act(dim, budget):
    # settings of numgeo-ACT (DEEM.optimize): fitness-focused sub-populations,
    # exponential reduction of 4 -> 2 sub-populations, damping-periodic bounds,
    # population shrinking to one half
    npart = int(min(60, max(24, 6 * dim)))
    return dict(nparticles_max=npart, nparticles_min=npart // 2, npop_max=4, npop_min=2,
                method_subswarm_reduction='exponential',
                method_subswarm_creation='fitness-focused',
                method_boundary='damping-periodic')


def _deem_published(dim, budget):
    return dict(nparticles_max=50, nparticles_min=50)


BASE_ALGORITHMS = {
    'deemi': ('DEEMI', _deemi_defaults),
    'deemi-act': ('DEEMI', _deemi_act),
    'deem': ('DEEM', _deem_published),
    'scipy-de': ('scipy', None),
}


def parse_algorithm(spec):
    """'[impl:]name[+key=value...]' -> (impl, name, overrides)."""
    impl = 'current'
    if ':' in spec.split('+')[0]:
        impl, spec = spec.split(':', 1)
    parts = spec.split('+')
    name, overrides = parts[0], {}
    for part in parts[1:]:
        key, value = part.split('=', 1)
        overrides[key] = ast.literal_eval(value)
    if name not in BASE_ALGORITHMS:
        raise ValueError(f'unknown algorithm {name!r}')
    return impl, name, overrides


def run_one(task):
    """Run one (algorithm, problem, dimension, regime, seed) task; returns a result row."""
    spec, name, dim, regime, seed = task
    budget = REGIMES[regime] * dim
    problem = make_problem(name, dim)
    recorder = BudgetRecorder(problem, budget)
    impl, algorithm, overrides = parse_algorithm(spec)
    family, builder = BASE_ALGORITHMS[algorithm]
    start = time.perf_counter()
    if family == 'scipy':
        from scipy.optimize import differential_evolution
        options = dict(popsize=15, polish=False, tol=0.0, atol=0.0, init='latinhypercube')
        options.update(overrides)
        npop = options['popsize'] * dim
        maxiter = int(math.ceil(budget / npop))     # (maxiter + 1) generations >= budget
        with contextlib.redirect_stdout(io.StringIO()):
            differential_evolution(recorder, list(zip(problem.lb, problem.ub)), maxiter=maxiter,
                                   seed=seed, **options)
    else:
        options = builder(dim, budget)
        overrides = dict(overrides)
        # pseudo-options of the harness: population size per dimension and the
        # final population as a fraction of the initial one
        np_per_dim = overrides.pop('np_per_dim', None)
        np_min_frac = overrides.pop('np_min_frac', None)
        if np_per_dim is not None:
            options['nparticles_max'] = options['nparticles_min'] = max(8, int(round(np_per_dim * dim)))
        if np_min_frac is not None:
            options['nparticles_min'] = max(4, int(round(np_min_frac * options['nparticles_max'])))
        options.update(overrides)
        npart = options['nparticles_max']
        mean_population = float(npart)
        shrink = options.get('nparticles_min', npart) / npart
        if shrink < 1.0:
            # exponential population shrinking: mean size npart * (r - 1) / ln r, so
            # that the planned generations use the budget (maxfev stops the run)
            mean_population = npart * (shrink - 1.0) / math.log(shrink)
        maxiter = int(math.ceil((budget - npart) / mean_population))
        options.setdefault('maxiter', maxiter)
        options.setdefault('maxfev', budget)
        with contextlib.redirect_stdout(io.StringIO()):
            if family == 'DEEMI':
                cls = _load(impl, 'DEEMI').DEEM
                optimizer = cls(recorder, problem.lb, problem.ub, seed=seed, **options)
            else:
                cls = _load(impl, 'DEEM').DEEM
                np.random.seed(seed)
                if 'npop_max' in options:
                    options['nswarm_max'] = options.pop('npop_max')
                if 'npop_min' in options:
                    options['nswarm_min'] = options.pop('npop_min')
                optimizer = cls(recorder, problem.lb, problem.ub, **options)
            optimizer.log.plot_results = lambda: None
            optimizer.update()
    elapsed = time.perf_counter() - start
    restarts = getattr(optimizer, 'n_restarts', '') if family != 'scipy' else ''
    row = dict(algorithm=spec, function=name, dim=dim, regime=regime, budget=budget, seed=seed,
               error=recorder.best, nfev=min(recorder.n, budget), time=round(elapsed, 3),
               restarts=restarts)
    for fraction in CHECKPOINTS:
        row[f'error_{int(fraction * 100)}'] = recorder.at.get(fraction, recorder.best)
    return row


def _worker_init(impls):
    IMPLEMENTATIONS.update(impls)
    os.chdir(tempfile.mkdtemp(prefix='deemi-validation-'))   # the DEEM logger writes to ./
    import warnings
    warnings.simplefilter('ignore')


def command_run(args):
    impls = {}
    for item in args.impl or []:
        label, location = item.split('=', 1)
        parent, package = location.rsplit(':', 1)
        impls[label] = (os.path.abspath(parent), package)
    IMPLEMENTATIONS.update(impls)
    suites = list(SUITES) if args.suite == 'all' else args.suite.split(',')
    regimes = list(REGIMES) if args.regime == 'all' else args.regime.split(',')
    dims = [int(d) for d in args.dims.split(',')]
    algorithms = args.algorithms.split(',')
    functions = [f for s in suites for f in SUITES[s]]
    if args.functions:
        functions = args.functions.split(',')
    for spec in algorithms:
        parse_algorithm(spec)
    done = set()
    if os.path.exists(args.out) and not args.overwrite:
        with open(args.out, newline='') as handle:
            for row in csv.DictReader(handle):
                done.add((row['algorithm'], row['function'], int(row['dim']), row['regime'], int(row['seed'])))
    tasks = [t for t in itertools.product(algorithms, functions, dims, regimes, range(args.seeds))
             if t not in done]
    # longest tasks first for a better load balance
    tasks.sort(key=lambda t: -REGIMES[t[3]] * t[2])
    print(f'{len(tasks)} runs ({len(done)} already in {args.out})')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fields = ['algorithm', 'function', 'dim', 'regime', 'budget', 'seed', 'error', 'nfev', 'time', 'restarts'] + \
             [f'error_{int(f * 100)}' for f in CHECKPOINTS]
    write_header = args.overwrite or not os.path.exists(args.out)
    with open(args.out, 'w' if args.overwrite else 'a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        start = time.time()
        with futures.ProcessPoolExecutor(max_workers=args.jobs, initializer=_worker_init,
                                         initargs=(impls,)) as pool:
            for count, row in enumerate(pool.map(run_one, tasks, chunksize=1), 1):
                writer.writerow(row)
                if count % 200 == 0 or count == len(tasks):
                    handle.flush()
                    print(f'{count}/{len(tasks)} runs, {time.time() - start:.0f} s', flush=True)


# =========================================================================== #
#  Report                                                                      #
# =========================================================================== #
def _load_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline='') as handle:
            for row in csv.DictReader(handle):
                row['dim'] = int(row['dim'])
                row['seed'] = int(row['seed'])
                row['error'] = float(row['error'])
                rows.append(row)
    return rows


def _suite_of(function):
    for suite, members in SUITES.items():
        if function in members:
            return suite
    return 'other'


def _fmt(value):
    if not np.isfinite(value):
        return 'inf'
    if value == 0.0:
        return '0'
    return f'{value:.2e}'


def command_report(args):
    from scipy.stats import mannwhitneyu, wilcoxon
    rows = _load_rows(args.results)
    algorithms = args.algorithms.split(',') if args.algorithms else sorted({r['algorithm'] for r in rows})
    reference = args.reference
    labels = dict(item.rsplit('=', 1) for item in args.labels.split(';') if item) if args.labels else {}

    def name(spec):
        return labels.get(spec, f'`{spec}`')
    data = {}
    for r in rows:
        if r['algorithm'] in algorithms:
            data.setdefault((r['function'], r['dim'], r['regime'], r['algorithm']), []).append(r['error'])
    configs = sorted({k[:3] for k in data}, key=lambda c: (_suite_of(c[0]), c[2], c[0], c[1]))
    os.makedirs(args.out, exist_ok=True)
    lines = []
    out = lines.append
    out('# DEEMI validation results\n')
    out(f'Seeds per configuration: {SEEDS} (0-{SEEDS - 1}); budgets: '
        + ', '.join(f'{k} = {v} x d' for k, v in REGIMES.items())
        + f'; success = error <= {TARGET:g}. Reference: {name(reference)}.\n')
    summary = []
    for suite in [s for s in SUITES if any(_suite_of(c[0]) == s for c in configs)]:
        for regime in REGIMES:
            cs = [c for c in configs if _suite_of(c[0]) == suite and c[2] == regime]
            if not cs:
                continue
            out(f'\n## {suite} set, {regime} regime\n')
            out('Median error [IQR] over the seeds; bold: best median.\n')
            out('| function | d | ' + ' | '.join(name(a) for a in algorithms) + ' |')
            out('|---|---|' + '---|' * len(algorithms))
            ranks = {a: [] for a in algorithms}
            for c in cs:
                medians = {}
                cells = []
                for a in algorithms:
                    v = np.asarray(data.get(c + (a,), []), dtype=float)
                    medians[a] = np.median(v) if v.size else np.nan
                best = np.nanmin(list(medians.values()))
                for a in algorithms:
                    v = np.asarray(data.get(c + (a,), []), dtype=float)
                    if not v.size:
                        cells.append('-')
                        continue
                    q1, q3 = np.percentile(v, [25, 75])
                    text = f'{_fmt(medians[a])} [{_fmt(q1)}, {_fmt(q3)}]'
                    cells.append(f'**{text}**' if medians[a] == best else text)
                out(f'| {c[0]} | {c[1]} | ' + ' | '.join(cells) + ' |')
                # ranks of all runs of this configuration (mean rank per algorithm)
                pooled = [(e, a) for a in algorithms for e in data.get(c + (a,), [])]
                values = np.array([p[0] for p in pooled])
                from scipy.stats import rankdata
                rk = rankdata(values)
                for a in algorithms:
                    sel = [rk[i] for i, p in enumerate(pooled) if p[1] == a]
                    if sel:
                        ranks[a].append(np.mean(sel))
            # summary versus the reference
            out('\n| algorithm | mean rank | success rate | wins / ties / losses vs reference '
                '| Wilcoxon p (vs reference) | median of log10 error ratio |')
            out('|---|---|---|---|---|---|')
            for a in [al for al in algorithms if ranks[al]]:
                succ = [np.mean(np.asarray(data[c + (a,)]) <= TARGET) for c in cs if c + (a,) in data]
                wins = ties = losses = 0
                ref_med, alg_med = [], []
                for c in cs:
                    if c + (a,) not in data or c + (reference,) not in data:
                        continue
                    x = np.asarray(data[c + (a,)]); y = np.asarray(data[c + (reference,)])
                    ref_med.append(np.log10(np.median(y) + 1e-12)); alg_med.append(np.log10(np.median(x) + 1e-12))
                    if a == reference:
                        continue
                    if np.array_equal(np.sort(x), np.sort(y)):
                        ties += 1
                        continue
                    p = mannwhitneyu(x, y, alternative='two-sided').pvalue
                    if p >= 0.05:
                        ties += 1
                    elif np.median(x) < np.median(y) or (np.median(x) == np.median(y) and np.mean(x) < np.mean(y)):
                        wins += 1
                    else:
                        losses += 1
                if a == reference:
                    wtl, pw, ratio = '-', '-', '-'
                else:
                    diff = np.array(alg_med) - np.array(ref_med)
                    if diff.size and np.any(diff != 0):
                        pw = f'{wilcoxon(alg_med, ref_med).pvalue:.3g}'
                    else:
                        pw = '1'
                    wtl = f'{wins} / {ties} / {losses}'
                    ratio = f'{np.median(diff):+.2f}' if diff.size else '-'
                out(f'| {name(a)} | {np.mean(ranks[a]):.2f} | {100 * np.mean(succ):.0f} % | {wtl} | {pw} | {ratio} |')
                summary.append((suite, regime, a, np.mean(ranks[a])))
    with open(os.path.join(args.out, 'report.md'), 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    try:
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        groups = sorted({(s, r) for s, r, _, _ in summary}, key=lambda g: (list(SUITES).index(g[0]) if g[0] in SUITES else 9, g[1]))
        fig, ax = plt.subplots(figsize=(1.6 + 1.3 * len(groups), 3.2))
        width = 0.8 / max(1, len(algorithms))
        for i, a in enumerate(algorithms):
            values = [next((m for s, r, al, m in summary if (s, r) == g and al == a), np.nan) for g in groups]
            ax.bar(np.arange(len(groups)) + (i - (len(algorithms) - 1) / 2) * width, values, width,
                   label=labels.get(a, a))
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels([f'{s}\n{r}' for s, r in groups], fontsize=8)
        ax.set_ylabel('mean rank (lower is better)')
        ax.legend(fontsize=7, frameon=False, ncol=2)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, args.figure), dpi=150)
        plt.close(fig)
    except Exception as exc:   # the tables are the result; the figure is optional
        print(f'figure skipped: {exc}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('#=~')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run', help='run the benchmark and append rows to a CSV file')
    run.add_argument('--suite', default='development,holdout', help="development, holdout, special or all")
    run.add_argument('--regime', default='all', help='expensive, large or all')
    run.add_argument('--dims', default=','.join(map(str, DIMENSIONS)))
    run.add_argument('--functions', default='', help='comma-separated subset of the functions')
    run.add_argument('--algorithms', default='deemi,deemi-act,deem,scipy-de')
    run.add_argument('--impl', action='append', help='LABEL=PARENT_DIRECTORY:PACKAGE (another DEEM copy)')
    run.add_argument('--seeds', type=int, default=SEEDS)
    run.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    run.add_argument('--out', default=os.path.join('validation', 'results.csv'))
    run.add_argument('--overwrite', action='store_true')
    rep = sub.add_parser('report', help='tables and a figure from one or more CSV files')
    rep.add_argument('results', nargs='+')
    rep.add_argument('--algorithms', default='')
    rep.add_argument('--reference', default='deemi')
    rep.add_argument('--out', default='validation')
    rep.add_argument('--figure', default='mean-rank.png')
    rep.add_argument('--labels', default='', help="display names: 'spec=Label;spec=Label'")
    args = parser.parse_args()
    if args.command == 'run':
        command_run(args)
    else:
        command_report(args)


if __name__ == '__main__':
    main()
