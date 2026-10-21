"""Elimination tests (Friedman, and Wilcoxon for two configurations) and rank
aggregation.

See MAPPING.md section E. Ports R_stuff/race.R:140-221 (aux2_friedman / aux_friedman)
and R_stuff/race.R:560-590 (get_ranks / overall_ranks). As in R, the race uses the
Friedman test while more than two configurations are alive and a paired Wilcoxon
signed-rank test once exactly two are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm, rankdata, t as student_t


@dataclass
class EliminationResult:
    alive: np.ndarray
    dropped_any: bool
    p_value: float


def get_ranks(y: np.ndarray) -> np.ndarray:
    """Ports get_ranks for the friedman case (race.R:560-561): rank each row
    (instance), sum ranks per column (configuration). `y` must have no NaNs."""
    r = rankdata(y, axis=1, method="average")
    return r.sum(axis=0)


def friedman_eliminate(results: np.ndarray, alive: np.ndarray, which_alive: np.ndarray,
                        conf_level: float) -> EliminationResult:
    """Ports aux2_friedman (race.R:140-179): Friedman omnibus test, and on
    rejection, a Conover post-hoc multiple-comparison to decide which
    configurations to keep (those not significantly worse than the best)."""
    y = results[:, which_alive]
    n, k = y.shape
    r = rankdata(y, axis=1, method="average")
    R = r.sum(axis=0)
    order = np.argsort(R)

    ties_sum = 0.0
    for row in r:
        _vals, counts = np.unique(row, return_counts=True)
        ties_sum += np.sum(counts**3 - counts)
    denom = n * k * (k + 1) - (ties_sum / (k - 1) if k > 1 else 0.0)

    new_alive = alive.copy()
    dropped_any = False
    p_value = 1.0

    if denom <= 0:
        # Every row is fully tied across all alive configurations: no evidence of a
        # difference, nothing to eliminate.
        return EliminationResult(alive=new_alive, dropped_any=False, p_value=1.0)

    statistic = (12.0 * np.sum((R - n * (k + 1) / 2.0) ** 2)) / denom
    p_value = float(chi2.sf(statistic, k - 1))

    alpha = 1 - conf_level
    if not math.isnan(p_value) and p_value < alpha:
        A = float(np.sum(r.astype(float) ** 2))
        df = (n - 1) * (k - 1)
        t_crit = student_t.ppf(1 - alpha / 2, df=df)
        spread = math.sqrt(max(0.0, (2 * (n * A - np.sum(R**2))) / df))
        crit = t_crit * spread

        keep_local = [order[0]]
        for j in range(1, k):
            if abs(R[order[j]] - R[order[0]]) > crit:
                break
            keep_local.append(order[j])
        keep_global = {which_alive[idx] for idx in keep_local}

        for idx in which_alive:
            if idx not in keep_global:
                new_alive[idx] = False
        dropped_any = True

    return EliminationResult(alive=new_alive, dropped_any=dropped_any, p_value=p_value)


def _signrank_counts(n: int) -> list:
    """counts[v] = number of subsets of {1, ..., n} summing to v: the exact null
    distribution of the signed-rank statistic V, times 2**n (R's dsignrank)."""
    counts = [1] + [0] * (n * (n + 1) // 2)
    for k in range(1, n + 1):
        for v in range(len(counts) - 1, k - 1, -1):
            counts[v] += counts[v - k]
    return counts


def wilcoxon_signed_rank_pvalue(diffs: np.ndarray) -> float:
    """Two-sided p-value of R's wilcox.test(x, y, paired = TRUE) with its defaults,
    given diffs = x - y (wilcox.test.default in R's stats package).

    Zero differences are dropped. The p-value is exact when fewer than 50 remain and
    none of their absolute values are tied; otherwise it uses the normal
    approximation with tie and continuity corrections.
    """
    x = diffs[diffs != 0]
    n = len(x)
    r = rankdata(np.abs(x), method="average")
    statistic = float(r[x > 0].sum())
    zeroes = n < len(diffs)
    ties = len(np.unique(r)) != n
    if n < 50 and not ties and not zeroes:
        counts = _signrank_counts(n)
        v = int(round(statistic))
        if statistic > n * (n + 1) / 4:
            p = sum(counts[v:]) / 2 ** n  # psignrank(V - 1, n, lower.tail = FALSE)
        else:
            p = sum(counts[:v + 1]) / 2 ** n  # psignrank(V, n)
        return min(2 * p, 1.0)
    _values, nties = np.unique(r, return_counts=True)
    z = statistic - n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - np.sum(nties ** 3 - nties) / 48)
    z = (z - np.sign(z) * 0.5) / sigma
    return float(2 * min(norm.cdf(z), norm.sf(z)))


def wilcoxon_eliminate(results: np.ndarray, alive: np.ndarray, which_alive: np.ndarray,
                       conf_level: float) -> EliminationResult:
    """Ports aux_friedman's branch for exactly two alive configurations
    (race.R:191-220).

    If one configuration is at least as good on every instance, the other is
    eliminated without a test; this includes the case where every result is tied,
    where the second one is dropped. Otherwise a paired Wilcoxon signed-rank test is
    run, and if it is significant the worse configuration by the pseudo-median of
    the differences (the median of the Walsh averages) is eliminated.
    """
    first, second = which_alive[0], which_alive[1]
    diffs = results[:, first] - results[:, second]
    p_value = 0.0
    dropped_any = True
    # Avoid the test if the answer is obvious.
    if np.all(diffs <= 0):
        worse = second
    elif np.all(diffs >= 0):
        worse = first
    else:
        p_value = wilcoxon_signed_rank_pvalue(diffs)
        if p_value >= 1 - conf_level:
            dropped_any = False
        nonzero = diffs[diffs != 0]
        i, j = np.triu_indices(len(nonzero))
        worse = second if np.median((nonzero[i] + nonzero[j]) / 2) <= 0 else first
    new_alive = alive.copy()
    if dropped_any:
        new_alive[worse] = False
    return EliminationResult(alive=new_alive, dropped_any=dropped_any, p_value=p_value)


def eliminate(results: np.ndarray, alive: np.ndarray, which_alive: np.ndarray,
              conf_level: float) -> EliminationResult:
    """Ports aux_friedman (race.R:181-221): the Friedman test while more than two
    configurations are alive, the Wilcoxon test once exactly two are."""
    if len(which_alive) > 2:
        return friedman_eliminate(results, alive, which_alive, conf_level)
    return wilcoxon_eliminate(results, alive, which_alive, conf_level)


def overall_ranks(results: np.ndarray, which_alive: np.ndarray) -> np.ndarray:
    """Ports overall_ranks (race.R:566-590): a configuration evaluated on more
    instances is ranked better; configurations evaluated on the same number of
    instances are ordered by Friedman rank-sum over the rows they all share.

    Survivors can have different counts even without capping: a race may stop
    part-way through the previous instances, where elites already have results
    and newly sampled configurations do not.
    """
    if len(which_alive) == 1:
        return np.array([1.0])
    used_rows = ~np.isnan(results).all(axis=1)
    x = results[used_rows][:, which_alive]
    n_instances = (~np.isnan(x)).sum(axis=0)
    ranks = np.full(len(which_alive), np.inf)
    last_r = 0.0
    # Iterate from the largest to the lowest number of instances.
    for k in sorted(set(n_instances.tolist()), reverse=True):
        confs = np.where(n_instances == k)[0]
        r = np.array([1.0])
        if len(confs) > 1:
            y = x[:, confs]
            r = get_ranks(y[~np.isnan(y).any(axis=1)])
        r = r + last_r
        last_r = float(r.max())
        ranks[confs] = r
    return ranks
