#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 Jan Machaček
#
# This file is part of DEEM, released under the BSD 3-Clause License.
# See the LICENSE file in the project root for the full license text.

"""
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#               DEEM - Differential Evolution with Elitism and Multi-populations
#                            Copyright (C) 2023-2025 Jan Machacek
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
#
# author: Jan Machacek, jan-machacek@outlook.com
# date: 23.06.2026
#
# Optional surrogate-assisted pre-screening for DEEM. Designed for expensive
# objectives such as the automatic calibration of constitutive soil models, where
# a single objective evaluation is a finite-element simulation of an element test
# (minutes to hours). The surrogate ranks the trial vectors of an iteration and only
# the most promising fraction (+ an exploration quota) is evaluated on the real
# objective; the remaining already-evaluated candidates keep their personal best.
#
# A k-nearest-neighbour feasibility classifier additionally biases evaluation away
# from regions that previously produced a penalty value (the Hypo-ISA model is not
# numerically stable for all parameter combinations, which makes the cost function
# discontinuous; cf. Machaček et al. 2025, Sec. 4).
#
# Dependency-light: numpy + scipy only. The surrogate is a regularised radial basis
# function (RBF) interpolant; a Gaussian-process or random-forest backend can be
# substituted by overriding _fit_regressor/_predict.
#
# History:
# 23.06.2026, J. Machacek - Initial version (its an attempt... we'll see if it works)
#=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~~=~=~
"""

import numpy as np


class RBFSurrogate:
    """
    Regularised multiquadric RBF interpolant f(x) ~ sum_i w_i * phi(||x - x_i||).

    Inputs are scaled to the unit box before fitting for numerical conditioning.
    """

    def __init__(self, LB, UB, ridge: float = 1e-8, max_points: int = 400):
        self.LB = np.asarray(LB, dtype=float)
        self.span = np.asarray(UB, dtype=float) - self.LB
        self.span[self.span == 0.0] = 1.0
        self.ridge = ridge
        self.max_points = max_points
        self.Xs = None
        self.w = None
        self.c = 1.0

    def _scale(self, X):
        return (np.asarray(X, dtype=float) - self.LB) / self.span

    @staticmethod
    def _phi(r, c):
        return np.sqrt(r * r + c * c)   # multiquadric

    def fit(self, X, f):
        X = np.asarray(X, dtype=float); f = np.asarray(f, dtype=float)
        if len(X) > self.max_points:                 # keep the most recent points
            X = X[-self.max_points:]; f = f[-self.max_points:]
        Xs = self._scale(X)
        n = len(Xs)
        if n < 3:
            self.Xs = Xs; self.w = None; self.f_mean = float(np.mean(f)) if n else 0.0
            return
        # pairwise distances
        d = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=2)
        self.c = max(np.median(d[d > 0]) if np.any(d > 0) else 1.0, 1e-6)
        A = self._phi(d, self.c) + self.ridge * np.eye(n)
        try:
            self.w = np.linalg.solve(A, f)
        except np.linalg.LinAlgError:
            self.w, *_ = np.linalg.lstsq(A, f, rcond=None)
        self.Xs = Xs
        self.f_mean = float(np.mean(f))

    def predict(self, X):
        Xs = self._scale(X)
        if self.Xs is None or self.w is None:
            return np.full(len(Xs), getattr(self, 'f_mean', 0.0))
        d = np.linalg.norm(Xs[:, None, :] - self.Xs[None, :, :], axis=2)
        return self._phi(d, self.c) @ self.w

    def novelty(self, X):
        """Distance (in scaled space) to the nearest observed point: an exploration score."""
        Xs = self._scale(X)
        if self.Xs is None or len(self.Xs) == 0:
            return np.full(len(Xs), np.inf)
        d = np.linalg.norm(Xs[:, None, :] - self.Xs[None, :, :], axis=2)
        return np.min(d, axis=1)


class SurrogateManager:
    """
    Surrogate-assisted evaluation controller for DEEM.

    Parameters
    ----------
    LB, UB : array-like
        Search-space bounds.
    eval_frac : float
        Fraction of the population evaluated on the real objective each iteration
        once the surrogate is trained.
    explore_frac : float
        Fraction (of the evaluated budget) reserved for the most novel candidates.
    min_train : int
        Number of observations required before pre-screening starts; until then all
        candidates are evaluated.
    refit_every : int
        Refit the surrogate every 'refit_every' new observations.
    penalty : float
        Objective value flagged as infeasible (used by the feasibility classifier).
    feasibility : bool
        Enable the kNN feasibility bias.
    """

    def __init__(self, LB, UB, eval_frac: float = 0.5, explore_frac: float = 0.3,
                 min_train: int = 40, refit_every: int = 10, penalty: float = 1e22,
                 feasibility: bool = True):
        self.reg = RBFSurrogate(LB, UB)
        self.LB = np.asarray(LB, dtype=float)
        self.span = np.asarray(UB, dtype=float) - self.LB
        self.span[self.span == 0.0] = 1.0
        self.eval_frac = eval_frac
        self.explore_frac = explore_frac
        self.min_train = min_train
        self.refit_every = refit_every
        self.penalty = penalty
        self.feasibility = feasibility
        self.X_obs, self.f_obs, self.feas = [], [], []
        self._since_fit = 0
        self.n_saved = 0          # bookkeeping: evaluations avoided

    # -- feasibility (kNN vote in scaled space) --------------------------------
    def _feasible_prob(self, X, k: int = 5):
        if not self.feasibility or len(self.X_obs) < k:
            return np.ones(len(X))
        Xs = (np.asarray(X, float) - self.LB) / self.span
        Obs = (np.asarray(self.X_obs, float) - self.LB) / self.span
        feas = np.asarray(self.feas, float)
        d = np.linalg.norm(Xs[:, None, :] - Obs[None, :, :], axis=2)
        nn = np.argsort(d, axis=1)[:, :k]
        return feas[nn].mean(axis=1)

    def select(self, optimizer, candidates):
        """
        Return the indices of the candidates to evaluate on the real objective.

        Candidates that have never been evaluated (initialized == False) are always
        evaluated so the population always carries valid personal bests.
        """
        n = len(candidates)
        idx_all = list(range(n))
        fresh = [i for i, cs in enumerate(candidates) if not getattr(cs, 'initialized', False)]

        if len(self.X_obs) < self.min_train:
            return idx_all   # warm-up: evaluate everything

        X = np.array([cs.x for cs in candidates], dtype=float)
        f_hat = self.reg.predict(X)
        nov = self.reg.novelty(X)
        feas_p = self._feasible_prob(X)
        # exploitation score: low predicted objective, penalised by infeasibility risk
        exploit_rank = np.argsort(f_hat + (1.0 - feas_p) * (np.nanmax(f_hat) - np.nanmin(f_hat) + 1.0))

        budget = max(len(fresh) + 1, int(np.ceil(self.eval_frac * n)))
        budget = min(budget, n)
        n_explore = int(np.floor(self.explore_frac * budget))
        n_exploit = budget - n_explore

        chosen = set(fresh)
        for i in exploit_rank:
            if len(chosen) >= n_exploit:
                break
            chosen.add(int(i))
        # exploration: most novel not yet chosen
        for i in np.argsort(-nov):
            if len(chosen) >= budget:
                break
            chosen.add(int(i))

        self.n_saved += (n - len(chosen))
        return sorted(chosen)

    def observe(self, evaluated_candidates):
        """Record evaluated candidates and refit the surrogate periodically."""
        for cs in evaluated_candidates:
            self.X_obs.append(np.asarray(cs.x, dtype=float))
            self.f_obs.append(float(cs.f))
            self.feas.append(0.0 if cs.f >= self.penalty else 1.0)
            self._since_fit += 1
        # cap memory
        if len(self.X_obs) > 5000:
            self.X_obs = self.X_obs[-5000:]; self.f_obs = self.f_obs[-5000:]; self.feas = self.feas[-5000:]
        if self._since_fit >= self.refit_every:
            mask = [ff < self.penalty for ff in self.f_obs]   # fit on feasible only
            Xf = [x for x, m in zip(self.X_obs, mask) if m]
            ff = [v for v, m in zip(self.f_obs, mask) if m]
            if len(Xf) >= 3:
                self.reg.fit(np.array(Xf), np.array(ff))
            self._since_fit = 0


# =====================================================================================
#  Second surrogate model: Gaussian Process (GP) with uncertainty-aware selection
# =====================================================================================
#
# History:
# 24.06.2026, J. Machacek - Added a Gaussian-process surrogate (GPSurrogate) and a
#                           matching controller (GPSurrogateManager). Unlike the RBF
#                           interpolant above, the GP returns a predictive *variance*,
#                           which is used for a Lower-Confidence-Bound (LCB) selection
#                           rule mean - kappa * std. This is the classic surrogate-
#                           assisted / Bayesian-optimisation acquisition: it evaluates
#                           candidates that are either promising (low predicted cost)
#                           or uncertain (poorly covered by observations), which suits
#                           the rugged, discontinuous calibration cost function.
#                           Dependency-light: numpy + scipy.linalg only.
# =====================================================================================

class GPSurrogate:
    """
    Gaussian-process regression with a squared-exponential (RBF) kernel and a
    constant mean. Inputs are scaled to the unit box; the length scale is set to
    the median pairwise distance (a robust, training-free heuristic). The model is
    fitted by a Cholesky factorisation of (K + sigma_n^2 I) with adaptive jitter.

        k(x, x') = sigma_f^2 * exp( -||x - x'||^2 / (2 l^2) )

    Provides predict() (posterior mean) and predict_std() (posterior std), the
    ingredients of the Lower-Confidence-Bound acquisition used by GPSurrogateManager.
    """

    def __init__(self, LB, UB, noise: float = 1e-6, max_points: int = 300):
        self.LB = np.asarray(LB, dtype=float)
        self.span = np.asarray(UB, dtype=float) - self.LB
        self.span[self.span == 0.0] = 1.0
        self.noise = float(noise)
        self.max_points = int(max_points)
        self.Xs = None            # scaled training inputs
        self.alpha = None         # K^-1 (y - mean)
        self.L = None             # Cholesky factor of K + noise I
        self.ell = 1.0            # length scale (scaled space)
        self.sig2 = 1.0           # signal variance
        self.y_mean = 0.0

    def _scale(self, X):
        return (np.asarray(X, dtype=float) - self.LB) / self.span

    def _kernel(self, A, B):
        # squared-exponential kernel matrix between row-sets A and B (scaled space)
        d2 = (np.sum(A * A, axis=1)[:, None]
              + np.sum(B * B, axis=1)[None, :]
              - 2.0 * A @ B.T)
        np.maximum(d2, 0.0, out=d2)
        return self.sig2 * np.exp(-0.5 * d2 / (self.ell * self.ell))

    def fit(self, X, f):
        X = np.asarray(X, dtype=float)
        f = np.asarray(f, dtype=float)
        if len(X) > self.max_points:                 # keep the most recent points
            X = X[-self.max_points:]
            f = f[-self.max_points:]
        Xs = self._scale(X)
        n = len(Xs)
        if n < 3:
            self.Xs = Xs if n else None
            self.alpha = self.L = None
            self.y_mean = float(np.mean(f)) if n else 0.0
            return

        # hyperparameters from data (training-free heuristics)
        d = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=2)
        pos = d[d > 0]
        self.ell = max(float(np.median(pos)) if pos.size else 1.0, 1e-3)
        self.sig2 = max(float(np.var(f)), 1e-12)
        self.y_mean = float(np.mean(f))

        y = f - self.y_mean
        K = self._kernel(Xs, Xs)
        # adaptive jitter: grow the diagonal until the Cholesky succeeds
        jitter = self.noise * self.sig2
        for _ in range(8):
            try:
                L = np.linalg.cholesky(K + (jitter + 1e-12) * np.eye(n))
                self.L = L
                self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                self.Xs = Xs
                return
            except np.linalg.LinAlgError:
                jitter *= 10.0
        # last resort: ridge least-squares (no usable variance)
        self.L = None
        self.alpha = np.linalg.lstsq(K + 1e-3 * np.eye(n), y, rcond=None)[0]
        self.Xs = Xs

    def predict(self, X):
        """Posterior mean."""
        Xs = self._scale(X)
        if self.Xs is None or self.alpha is None:
            return np.full(len(Xs), self.y_mean)
        Ks = self._kernel(Xs, self.Xs)
        return self.y_mean + Ks @ self.alpha

    def predict_std(self, X):
        """Posterior standard deviation (exploration signal)."""
        Xs = self._scale(X)
        if self.Xs is None or self.L is None:
            # no fitted covariance -> treat everything as maximally uncertain
            return np.full(len(Xs), np.sqrt(self.sig2))
        Ks = self._kernel(Xs, self.Xs)                # (m, n)
        v = np.linalg.solve(self.L, Ks.T)             # (n, m)
        var = self.sig2 - np.sum(v * v, axis=0)
        return np.sqrt(np.maximum(var, 0.0))

    def novelty(self, X):
        """Distance to the nearest observed point (scaled space)."""
        Xs = self._scale(X)
        if self.Xs is None or len(self.Xs) == 0:
            return np.full(len(Xs), np.inf)
        d = np.linalg.norm(Xs[:, None, :] - self.Xs[None, :, :], axis=2)
        return np.min(d, axis=1)


class GPSurrogateManager:
    """
    Surrogate-assisted evaluation controller backed by a Gaussian process.

    Selection uses the Lower-Confidence-Bound acquisition

        LCB(x) = mean(x) - kappa * std(x)            (minimisation)

    so the real objective is spent on candidates that are promising (low mean) or
    uncertain (high std). A larger `kappa` favours exploration. This is the classic
    surrogate-assisted DE / Bayesian-optimisation rule and is complementary to the
    RBF-interpolant SurrogateManager above. The same kNN feasibility bias steers
    evaluations away from regions that previously returned the penalty value.

    The interface matches SurrogateManager, so it is a drop-in replacement:
    pass an instance as `DEEM(..., surrogate=GPSurrogateManager(LB, UB))`.

    Parameters
    ----------
    LB, UB : array-like
        Search-space bounds.
    eval_frac : float
        Fraction of the population evaluated on the real objective each iteration.
    explore_frac : float
        Fraction of the evaluated budget reserved for the most *uncertain* (highest
        posterior std) candidates.
    kappa : float
        Exploration weight of the LCB acquisition.
    min_train : int
        Observations required before pre-screening starts (warm-up evaluates all).
    refit_every : int
        Refit the GP every `refit_every` new observations.
    penalty : float
        Objective value flagged as infeasible (feasibility classifier).
    feasibility : bool
        Enable the kNN feasibility bias.
    """

    def __init__(self, LB, UB, eval_frac: float = 0.5, explore_frac: float = 0.3,
                 kappa: float = 1.5, min_train: int = 40, refit_every: int = 10,
                 penalty: float = 1e22, feasibility: bool = True):
        self.reg = GPSurrogate(LB, UB)
        self.LB = np.asarray(LB, dtype=float)
        self.span = np.asarray(UB, dtype=float) - self.LB
        self.span[self.span == 0.0] = 1.0
        self.eval_frac = eval_frac
        self.explore_frac = explore_frac
        self.kappa = kappa
        self.min_train = min_train
        self.refit_every = refit_every
        self.penalty = penalty
        self.feasibility = feasibility
        self.X_obs, self.f_obs, self.feas = [], [], []
        self._since_fit = 0
        self.n_saved = 0

    def _feasible_prob(self, X, k: int = 5):
        if not self.feasibility or len(self.X_obs) < k:
            return np.ones(len(X))
        Xs = (np.asarray(X, float) - self.LB) / self.span
        Obs = (np.asarray(self.X_obs, float) - self.LB) / self.span
        feas = np.asarray(self.feas, float)
        d = np.linalg.norm(Xs[:, None, :] - Obs[None, :, :], axis=2)
        nn = np.argsort(d, axis=1)[:, :k]
        return feas[nn].mean(axis=1)

    def select(self, optimizer, candidates):
        """
        Return the indices of the candidates to evaluate on the real objective.

        Candidates that have never been evaluated (initialized == False) are always
        evaluated so the population always carries valid personal bests.
        """
        n = len(candidates)
        idx_all = list(range(n))
        fresh = [i for i, cs in enumerate(candidates) if not getattr(cs, 'initialized', False)]

        if len(self.X_obs) < self.min_train:
            return idx_all   # warm-up: evaluate everything

        X = np.array([cs.x for cs in candidates], dtype=float)
        mean = self.reg.predict(X)
        std = self.reg.predict_std(X)
        feas_p = self._feasible_prob(X)

        # Lower-Confidence-Bound acquisition (lower is better), with an
        # infeasibility penalty scaled to the spread of the acquisition values.
        lcb = mean - self.kappa * std
        spread = (np.nanmax(lcb) - np.nanmin(lcb) + 1.0)
        acq = lcb + (1.0 - feas_p) * spread
        exploit_rank = np.argsort(acq)

        budget = max(len(fresh) + 1, int(np.ceil(self.eval_frac * n)))
        budget = min(budget, n)
        n_explore = int(np.floor(self.explore_frac * budget))
        n_exploit = budget - n_explore

        chosen = set(fresh)
        for i in exploit_rank:
            if len(chosen) >= n_exploit:
                break
            chosen.add(int(i))
        # exploration: evaluate the most *uncertain* candidates not yet chosen
        for i in np.argsort(-std):
            if len(chosen) >= budget:
                break
            chosen.add(int(i))

        self.n_saved += (n - len(chosen))
        return sorted(chosen)

    def observe(self, evaluated_candidates):
        """Record evaluated candidates and refit the GP periodically."""
        for cs in evaluated_candidates:
            self.X_obs.append(np.asarray(cs.x, dtype=float))
            self.f_obs.append(float(cs.f))
            self.feas.append(0.0 if cs.f >= self.penalty else 1.0)
            self._since_fit += 1
        if len(self.X_obs) > 5000:
            self.X_obs = self.X_obs[-5000:]; self.f_obs = self.f_obs[-5000:]; self.feas = self.feas[-5000:]
        if self._since_fit >= self.refit_every:
            mask = [ff < self.penalty for ff in self.f_obs]   # fit on feasible only
            Xf = [x for x, m in zip(self.X_obs, mask) if m]
            ff = [v for v, m in zip(self.f_obs, mask) if m]
            if len(Xf) >= 3:
                self.reg.fit(np.array(Xf), np.array(ff))
            self._since_fit = 0
