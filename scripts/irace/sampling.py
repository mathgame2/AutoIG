"""Uniform, Sobol, and model-based (truncated-normal) sampling of configurations.

See MAPPING.md section B. Ports the numeric-only branches of:
  - R_stuff/generation.R: generate_sobol/sampleSobol, generate_uniform/sampleUniform,
    sample_from_model/sampleModel
  - R_stuff/parameters.R: sample_unif.ParamInt/ParamReal, sample_model.ParamInt/ParamReal,
    param_quantile.ParamInt/ParamReal
  - pseudocode/util_functions.txt: sampleUniform(), sample()

Real values are rounded to `Parameter.digits` decimal places everywhere, as R does
(parameters.R:639,655,675). An optional user `repair` function is then applied to
every new configuration, as R's repairConfigurations (generation.R:18-28).
"""

from __future__ import annotations

import math
import random

import numpy as np
from scipy.stats import qmc, truncnorm

from scenario import Configuration, Parameter

# sampleModel's forbidden-configuration retry limit (generation.R:277-292).
MAX_SAMPLING_RETRIES = 100


def sample_uniform_value(p: Parameter):
    if p.kind == "int":
        # sample_unif.ParamInt (parameters.R:588-600): +1 for correct rounding
        # before floor(), so every integer in [lower, upper] is equally likely.
        v = random.uniform(p.lower, p.upper + 1)
        return int(min(p.upper, math.floor(v)))
    return round(random.uniform(p.lower, p.upper), p.digits)


def repair_values(values: dict, params: list, repair) -> dict:
    """Ports repairConfigurations (generation.R:18-28) for one configuration: call
    the user's `repair(values, params)` on a copy of the values and use what it
    returns. Like R, the returned values are not checked against the domains, but
    there must be one for every parameter and no others."""
    if repair is None:
        return values
    repaired = repair(dict(values), params)
    names = [p.name for p in params]
    if not isinstance(repaired, dict) or set(repaired) != set(names):
        raise ValueError("'repair_configuration' must return a dict with a value for every "
                         f"parameter ({', '.join(names)}), got {repaired!r}")
    return {name: repaired[name] for name in names}


def sample_uniform_configs(params: list, n: int, next_id: int, repair=None):
    """Ports sampleUniform (generation.R:195-216), numeric branch only.

    Returns (configs, next_id) so the caller can thread a global id counter.
    """
    configs = []
    cid = next_id
    for _ in range(n):
        values = repair_values({p.name: sample_uniform_value(p) for p in params}, params, repair)
        configs.append(Configuration(id=cid, values=values, parent_id=None))
        cid += 1
    return configs, cid


def param_quantile(p: Parameter, u: float):
    """Ports param_quantile.ParamInt/ParamReal (parameters.R:603-677): map u in [0, 1)
    to a parameter value."""
    if p.kind == "int":
        return int(min(p.upper, math.floor(u * (p.upper + 1 - p.lower) + p.lower)))
    return min(p.upper, max(p.lower, round(u * (p.upper - p.lower) + p.lower, p.digits)))


def _sobol_values(params: list, n: int, repair=None) -> list:
    """Ports generate_sobol (generation.R:90-137), numeric branch: the first `n`
    points of a scrambled Sobol sequence, mapped through param_quantile and
    repaired, with duplicates removed (generation.R:143)."""
    m = max(0, math.ceil(math.log2(n)))
    sampler = qmc.Sobol(d=len(params), scramble=True, rng=np.random.default_rng(random.getrandbits(32)))
    points = sampler.random_base2(m)[:n]
    seen = set()
    unique = []
    for row in points:
        values = repair_values({p.name: param_quantile(p, float(u)) for p, u in zip(params, row)},
                               params, repair)
        key = tuple(values[name] for name in sorted(values))
        if key not in seen:
            seen.add(key)
            unique.append(values)
    return unique


def sample_sobol_configs(params: list, n: int, next_id: int, repair=None):
    """Ports sampleSobol (generation.R:140-159), numeric branch only (no forbidden
    filter). Duplicates are removed, so fewer than `n` configurations may be
    returned when the parameter space is small, as in R.

    Returns (configs, next_id) so the caller can thread a global id counter.
    """
    values_list = _sobol_values(params, n, repair)
    have = len(values_list)
    if have < n:
        needed = max(math.ceil(n + (n - have) * (2 - have / n)), min(len(params) * 5, 100))
        values_list = _sobol_values(params, needed, repair)[:n]
    configs = [Configuration(id=next_id + i, values=values, parent_id=None)
               for i, values in enumerate(values_list)]
    return configs, next_id + len(configs)


def sample_trunc_norm(mean: float, std: float, lower: float, upper: float, is_int: bool,
                      digits: int = 15):
    """Ports sample_model.ParamInt/ParamReal (parameters.R), i.e. the truncated-normal
    sampling described in pseudocode/util_functions.txt's sample(). Real values are
    rounded to `digits` (parameters.R:639)."""
    if std < 1e-12:
        if is_int:
            return int(round(min(upper, max(lower, mean))))
        return round(min(upper, max(lower, mean)), digits)

    m, u = (mean + 0.5, upper + 1) if is_int else (mean, upper)
    a, b = (lower - m) / std, (u - m) / std
    x = truncnorm.rvs(a, b, loc=m, scale=std)
    if is_int:
        return int(min(upper, math.floor(x)))
    return min(upper, max(lower, round(float(x), digits)))


def elite_weight(n_elites: int, rank: int) -> float:
    """Ports extractElites' `.WEIGHT.` formula (irace.R:435); rank is 1-indexed."""
    return (n_elites - rank + 1) / (n_elites * (n_elites + 1) / 2)


def shrink_std(std: float, n_params: int, n_new: int) -> float:
    """Ports the std-shrink factor used by updateModel (model.R:65) /
    pseudocode/util_functions.txt's sample()."""
    return std * (1.0 / n_new) ** (1.0 / n_params)


def _sample_one_from_model(params: list, elites: list, weights: list, model: dict, repair=None):
    parent = random.choices(elites, weights=weights, k=1)[0]
    values = {}
    for p in params:
        std, mean = model[p.name][parent.id]
        values[p.name] = sample_trunc_norm(mean, std, p.lower, p.upper, p.kind == "int", p.digits)
    return repair_values(values, params, repair), parent.id


def sample_from_model(params: list, elites: list, model: dict, n_new: int, next_id: int,
                      forbidden: set | None = None, repair=None):
    """Ports sample_from_model/sampleModel (generation.R:218-294), numeric branch only.

    Each new configuration picks a parent elite with probability proportional to
    `.WEIGHT.`, then samples every parameter from the parent's per-parameter
    (std, value) model entry via a truncated normal.

    `forbidden` is a set of Configuration.key() tuples that must not be produced:
    configurations previously rejected for returning Inf, which R adds to its
    forbidden list (race.R:1042-1045). Like sampleModel, filtered-out samples are
    replaced, giving up after MAX_SAMPLING_RETRIES rounds.

    `repair` is applied to each sample before the forbidden check, as in R.
    """
    if n_new <= 0:
        return [], next_id
    forbidden = forbidden or set()
    weights = [e.weight for e in elites]
    sampled = []
    for _ in range(MAX_SAMPLING_RETRIES + 1):
        while len(sampled) < n_new:
            sampled.append(_sample_one_from_model(params, elites, weights, model, repair))
        sampled = [(v, pid) for v, pid in sampled
                   if tuple(v[name] for name in sorted(v)) not in forbidden]
        if len(sampled) == n_new:
            break
    else:
        raise RuntimeError("irace tried 100 times to sample from the model a configuration not "
                           "forbidden without success, perhaps your constraints are too strict?")
    configs = [Configuration(id=next_id + i, values=values, parent_id=pid)
               for i, (values, pid) in enumerate(sampled)]
    return configs, next_id + n_new
