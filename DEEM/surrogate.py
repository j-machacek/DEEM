#!/usr/bin/env python3
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
