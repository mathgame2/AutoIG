"""Post-selection race: spend the budget left at the end of irace re-racing the
elites of every iteration, and return the winners.

See MAPPING.md section H. Ports R_stuff/psRace.R as called from irace_finish
(irace.R:659-672), i.e. `psRace(iraceResults, max_experiments = remainingBudget,
iteration_elites = TRUE)`, for the supported scope (non-deterministic, maxExperiments
budget, blockSize 1).
"""

from __future__ import annotations

import dataclasses

import numpy as np

import race
from report import Timer, irace_note, print_elapsed, print_elite_configurations

# truncate_conf_needs (psRace.R:90-93): at most this many configurations are
# considered when choosing which ones to race.
MAX_COMBINATION_SIZE = 16


def _combinations(codes: list, n: int) -> np.ndarray:
    """Bit i of each code selects candidate i (psRace.R:80, 87)."""
    codes = np.asarray(codes, dtype=np.int64)
    return ((codes[:, None] >> np.arange(n)) & 1).astype(bool)


def generate_combs_2(n: int) -> np.ndarray:
    """Ports generate_combs_2 (psRace.R:74-81): every subset of at least two of `n`
    candidates, in R's order (odd codes first, then even codes that are not powers
    of 2)."""
    max_pow = 2 ** n
    codes = list(range(3, max_pow, 2))
    if max_pow >= 8:
        powers = {2 ** i for i in range(3, n)}
        codes += [z for z in range(6, max_pow - 1, 2) if z not in powers]
    return _combinations(codes, n)


def generate_combs_1(n: int) -> np.ndarray:
    """Ports generate_combs_1 (psRace.R:83-88): every non-empty subset of `n`
    candidates."""
    return _combinations(list(range(1, 2 ** n)), n)


def _best_combination(combs: np.ndarray, needs: np.ndarray, max_experiments: int):
    """The feasible combination with the most candidates; ties go to the first one
    in `combs`, as which.max does (psRace.R:156-161, 181-186)."""
    feasible = np.flatnonzero(combs @ needs <= max_experiments)
    if feasible.size == 0:
        return None
    counts = combs[feasible].sum(axis=1)
    return combs[feasible[np.argmax(counts)]]


def select_configurations(candidates: list, needs: dict, max_experiments: int,
                          each_test: int, report=None) -> list:
    """Ports get_confs_for_psrace (psRace.R:95-197), non-deterministic branch.

    `candidates` are configuration ids in priority order and `needs[id]` is the
    number of already-seen instances that configuration has no result for. Returns
    the ids to race, in candidate order; fewer than two means there is nothing to
    race.

    `report(ids, needs)` is called with the selection and the pending instances R
    reports for it (report_selected, psRace.R:98-101).
    """
    report = report or (lambda ids, needs: None)
    pending = [(cid, needs[cid]) for cid in candidates if needs[cid] <= max_experiments]
    if len(pending) <= 1:
        return [cid for cid, _ in pending]

    # We want to evaluate on at least n_new more instances.
    n_new = each_test
    n_confs = max_experiments // n_new
    done = [cid for cid, n in pending if n == 0]
    if n_confs > 1 and len(done) >= n_confs:
        report(done[:n_confs], [0] * n_confs)
        return done[:n_confs]

    # Let's try first to evaluate on new instances.
    first = pending[:MAX_COMBINATION_SIZE]
    needs_new = np.array([n + n_new for _, n in first])
    winner = _best_combination(generate_combs_2(len(first)), needs_new, max_experiments)
    if winner is not None:
        chosen = [(cid, n + n_new) for (cid, n), keep in zip(first, winner) if keep]
        report([cid for cid, _ in chosen], [n for _, n in chosen])
        return [cid for cid, _ in chosen]

    # Not enough budget to see a new instance: keep every configuration evaluated on
    # all instances, and complete as many of the others as the budget allows.
    partial = [(cid, n) for cid, n in pending if n > 0][:MAX_COMBINATION_SIZE]
    if not partial:
        report(done, [0] * len(done))
        return done
    winner = _best_combination(generate_combs_1(len(partial)),
                               np.array([n for _, n in partial]), max_experiments)
    chosen = [(cid, 0) for cid in done] + [(cid, n) for (cid, n), keep in zip(partial, winner) if keep]
    report([cid for cid, _ in chosen], [n for _, n in chosen])
    return [cid for cid, _ in chosen]


def candidate_ids(all_elites: list, all_config_ids: list, rejected_ids: set) -> list:
    """psRace.R:105-112 with iteration_elites = TRUE: the elites of every iteration,
    latest first, then every other configuration in generation order, without
    rejected ones."""
    ordered = [cid for ids in reversed(all_elites) for cid in ids] + list(all_config_ids)
    return [cid for cid in dict.fromkeys(ordered) if cid not in rejected_ids]


def missing_counts(global_results: dict, ids: list) -> dict:
    """colCounts(experiments, value = NA): for each id, the number of evaluated
    instances with no result for it."""
    counts = dict.fromkeys(ids, 0)
    for bucket in global_results.values():
        for cid in bucket:
            if cid in counts:
                counts[cid] += 1
    return {cid: len(global_results) - n for cid, n in counts.items()}


def post_selection(scenario, pool, global_results: dict, all_configs: dict,
                   all_elites: list, rejected_ids: set, max_experiments: int,
                   min_survival: int, seed: int, executor=None, verbose: bool = True,
                   timer: Timer | None = None):
    """Ports psRace (psRace.R:42-297). Returns (elites, experiments_used), or None
    when there are not two configurations to race; the caller then keeps its
    current elites, as psRace returns the last iteration's elites (psRace.R:222-226).
    """
    timer = timer or Timer()
    if verbose:
        irace_note("Starting post-selection:\n")

    def report_selected(ids, conf_needs):
        if verbose:
            print(f"# Configurations selected: {', '.join(map(str, ids))}.\n"
                  f"# Pending instances: {', '.join(map(str, conf_needs))}.", flush=True)

    candidates = candidate_ids(all_elites, all_configs, rejected_ids)
    needs = missing_counts(global_results, candidates)
    selected = select_configurations(candidates, needs, max_experiments, scenario.each_test,
                                     report=report_selected)
    if len(selected) < 2:
        if verbose and selected:
            # psRace.R:118-121.
            print(f"# Insufficient budget to race more than one configuration: {selected[0]} !")
        return None

    # psRace.R:238: configurations in generation order, not selection order.
    configs = [all_configs[cid] for cid in sorted(selected)]
    # psRace.R:241-243: new instances for the race, then race from the old ones.
    pool.generate(max_experiments / len(configs))
    n_old = len(global_results)
    if verbose:
        irace_note(f"seed: {seed}\n# Configurations: {len(configs)}\n"
                   f"# Available experiments: {max_experiments}\n# minSurvival: 1\n")

    # psRace.R:250-262: elitist racing with no elitist limit, minSurvival 1, no new
    # instances first and the first test after every old instance.
    result = race.elitist_race(
        configs, dataclasses.replace(scenario, elitist_limit=0, elitist=True), pool,
        global_results, next_instance=n_old + 1, elitist_new_instances=0,
        max_exp=max_experiments, min_survival=1, first_race=False, first_test=n_old,
        executor=executor, verbose=verbose, debug_level=scenario.debug_level)
    elites = race.extract_elites(result.ranked_configs, min_survival)
    if verbose:
        # psRace.R:266-277.
        print_elite_configurations(elites, scenario.parameters, scenario.debug_level)
        print_elapsed(timer)
    return elites, result.experiments_used
