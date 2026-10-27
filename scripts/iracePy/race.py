"""The elitist race: evaluate candidates instance-by-instance, eliminating
statistically-dominated candidates, reusing elites' historical results.

See MAPPING.md section D. Ports:
  - pseudocode/race.txt (the overall shape of the loop)
  - R_stuff/race.R:622-1074 (elitist_race)
  - R_stuff/race_state.R:234-278 (elitist_init_instances / no_elitist_init_instances)
  - R_stuff/race.R:560-599 (get_ranks/overall_ranks/update_is_elite)
  - R_stuff/irace.R:417-439 (extractElites)

Results for every (instance id, configuration id) pair ever computed are kept in a
single persistent `global_results` dict (instance_id -> {config_id: cost}), shared
across the whole irace_run. This plays the role of R's merged `iraceResults$experiments`
matrix, but keyed by real instance/config ids instead of positional row/column
indices, which sidesteps having to reconcile a race-local task-visiting order
(which reorders/shuffles "past" instances) against the global instance numbering.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np

from report import (irace_note, race_print_configurations, race_print_footer, race_print_header,
                    race_print_task)
from scenario import Configuration, Scenario
from stats import eliminate, get_ranks, overall_ranks
from target_runner import run_experiment


@dataclass
class RaceResult:
    ranked_configs: list  # Configuration objects, best first
    experiments_used: int
    break_reason: str
    tasks_processed: int  # number of race-local instance "tasks" reached
    # Configurations immediately rejected for returning Inf (race.R:887-905).
    rejected: list = field(default_factory=list)


def build_race_instance_order(next_instance: int, max_instances: int, elitist_new_instances: int,
                               sample_instances: bool, first_race: bool) -> list:
    """Ports elitist_init_instances (race_state.R:242-278)."""
    if first_race or next_instance == 1:
        return list(range(1, max_instances + 1))

    last_new = next_instance - 1 + elitist_new_instances
    if last_new > max_instances:
        last_new = max_instances
    new_instances = list(range(next_instance, last_new + 1)) if elitist_new_instances > 0 else []

    past_instances = list(range(1, next_instance))
    if sample_instances:
        random.shuffle(past_instances)

    if last_new + 1 <= max_instances:
        future_instances = list(range(last_new + 1, max_instances + 1))
        return new_instances + past_instances + future_instances
    return new_instances + past_instances


def no_elitist_race_instance_order(next_instance: int, max_instances: int) -> list:
    """Ports no_elitist_init_instances (race_state.R:234-242): the first race sees
    every instance; later races start at the first instance not yet seen."""
    if next_instance == 1:
        return list(range(1, max_instances + 1))
    return list(range(next_instance, max_instances + 1))


def execute_experiments(executor, scenario: Scenario, pool, inst_id: int, cfgs: list) -> list:
    """Ports execute_experiments (race-wrapper.R:510-600), local branches only: run
    every configuration in `cfgs` on pool entry `inst_id` and return their
    (cost, time) pairs in the same order.

    The target runner gets the instance's index in `scenario.instances`, not the
    pool id, as R's createExperimentList does (race.R:52-55, 67-71).

    With `executor=None` the experiments run one after another (R's lapply branch).
    Otherwise they are submitted to the executor (R's mclapply branch with load
    balancing). The first failure cancels the experiments not yet started and is
    re-raised, so a target-runner error still stops the whole run.
    """
    jobs = [(scenario.target_runner, cfg.id, pool.instance_id_for(inst_id),
             pool.seed_for(inst_id), pool.problem_for(inst_id), cfg.values) for cfg in cfgs]
    # run_target_runner (race-wrapper.R:406-440): each parameter's switch and the
    # scenario's targetCmdline build the command line; it runs in execDir
    # (race-wrapper.R:515-518).
    command_line = dict(switches={p.name: p.switch for p in scenario.parameters},
                        target_cmdline=scenario.target_cmdline, exec_dir=scenario.exec_dir)
    if executor is None:
        return [run_experiment(*job, **command_line) for job in jobs]
    futures = [executor.submit(run_experiment, *job, **command_line) for job in jobs]
    try:
        return [f.result() for f in futures]
    except BaseException:
        for f in futures:
            f.cancel()
        raise


def elitist_race(configs: list, scenario: Scenario, pool, global_results: dict,
                  next_instance: int, elitist_new_instances: int, max_exp: int,
                  min_survival: int, first_race: bool, verbose: bool = False,
                  executor=None, first_test: int | None = None,
                  debug_level: int = 0) -> RaceResult:
    """Ports elitist_race (race.R:633-1074). With `verbose`, prints R's race
    table: a header, one row per instance (race.R:804, 979) and a footer. With
    debug_level >= 2 (not in R), each row is followed by every configuration
    that was alive on that instance, its scores, and what happened to it."""
    show_configurations = verbose and debug_level >= 2

    def print_configurations(task, inst_id, task_alive, executed, status):
        """The configurations alive at the start of `task`, best rank first, then
        the rejected ones."""
        ranked = task_alive[np.isfinite(Results[:task + 1, task_alive]).all(axis=0)]
        rank_sums = dict(zip(ranked, get_ranks(Results[:task + 1, ranked]))) if ranked.size else {}
        order = sorted(task_alive, key=lambda i: rank_sums.get(i, np.inf))
        race_print_configurations(inst_id, task + 1, [
            (configs[i].id, i in executed, Results[task, i], np.mean(Results[:task + 1, i]),
             rank_sums.get(i), status.get(i, "alive")) for i in order])
    # R's firstTest argument (race.R:639): the post-selection race overrides it.
    first_test = scenario.first_test if first_test is None else first_test
    n = len(configs)
    alive = np.ones(n, dtype=bool)
    elitist = scenario.elitist

    if elitist:
        n_old_instances = 0 if first_race else next_instance - 1
        pool.ensure(max(pool.size, n_old_instances + elitist_new_instances))
        race_instances = build_race_instance_order(
            next_instance, pool.size, elitist_new_instances, scenario.sample_instances, first_race)
        elite_safe = 0 if first_race else elitist_new_instances + n_old_instances
    else:
        # Non-elitist racing (race.R:706-709): only instances not seen yet, and no
        # previous results (elite_data is NULL), so no configuration is elite.
        pool.ensure(max(pool.size, next_instance))
        race_instances = no_elitist_race_instance_order(next_instance, pool.size)
        elite_safe = 0

    # is_elite[i]: number of still-"safe" (already-evaluated) instances remaining
    # for configuration i among this race's first `elite_safe` tasks.
    is_elite = np.zeros(n, dtype=int)
    for i, cfg in enumerate(configs):
        count = 0
        for inst_id in race_instances[:elite_safe]:
            if cfg.id in global_results.get(inst_id, {}):
                count += 1
        is_elite[i] = count

    Results = np.full((len(race_instances), n), np.nan)
    for row, inst_id in enumerate(race_instances):
        bucket = global_results.get(inst_id) if elitist else None
        if not bucket:
            continue
        for i, cfg in enumerate(configs):
            if cfg.id in bucket:
                Results[row, i] = bucket[cfg.id]

    is_rejected = np.zeros(n, dtype=bool)

    def all_elite_instances_evaluated(task: int) -> bool:
        """Ports all_elite_instances_evaluated (race.R:694-704): every instance row
        seen so far or planned as an elite instance (R's Results has
        max(elite_safe, task) rows at this point) has a result for some alive
        configuration, and each alive configuration's number of results is a
        multiple of each_test. Always true in non-elitist racing (race.R:708)."""
        if not elitist or next_instance == 1:
            return True
        evaluated = ~np.isnan(Results[:max(elite_safe, task)][:, alive])
        return bool(evaluated.any(axis=1).all()
                    and np.all(evaluated.sum(axis=0) % scenario.each_test == 0))

    experiments_used = 0
    no_elimination = 0
    break_reason = None
    tasks_processed = len(race_instances)
    # Index of the best configuration in the last task (R's `best`).
    best = None

    if verbose:
        race_print_header()

    for task, inst_id in enumerate(race_instances):
        which_alive = np.where(alive)[0]
        nb_alive = len(which_alive)
        which_exe = which_alive[np.isnan(Results[task, which_alive])]

        if which_exe.size == 0:
            # Nothing new to run on this instance: everyone alive already has a
            # historical value for it (only possible while task < elite_safe).
            not_executed = which_alive[is_elite[which_alive] > 0]
            is_elite[not_executed] -= 1
            if task == 0:
                # race.R:787-797: the first task needs a best of its own.
                best = which_alive[0] if nb_alive == 1 else \
                    which_alive[np.argmin(get_ranks(Results[:1, which_alive]))]
            if verbose:
                now = time.time()
                race_print_task(".", Results[:task + 1], inst_id, task + 1, which_alive,
                                configs[best].id, best, experiments_used, now, now)
            if show_configurations:
                print_configurations(task, inst_id, which_alive, set(), {})
            continue

        # race.R:815: the stop checks only apply once every new instance (and every
        # previous instance) has been evaluated by some alive configuration.
        if all_elite_instances_evaluated(task):
            if nb_alive <= min_survival:
                break_reason = (f"number of alive configurations ({nb_alive}) <= minimum number "
                                f"of configurations ({min_survival})")
                tasks_processed = task
                break
            if (task + 1) > first_test and task % scenario.each_test == 0 and \
                    experiments_used + which_exe.size * scenario.each_test > max_exp:
                break_reason = (f"experiments for next test "
                                f"({experiments_used + which_exe.size * scenario.each_test}) "
                                f"> max experiments ({max_exp})")
                tasks_processed = task
                break
            if elitist and scenario.elitist_limit != 0 and no_elimination >= scenario.elitist_limit:
                break_reason = f"tests without elimination ({no_elimination}) >= elitistLimit ({scenario.elitist_limit})"
                tasks_processed = task
                break

        task_alive = which_alive
        start_time = time.time()
        outputs = execute_experiments(executor, scenario, pool, inst_id,
                                      [configs[i] for i in which_exe])
        for i, (cost, _time) in zip(which_exe, outputs):
            Results[task, i] = cost
            global_results.setdefault(inst_id, {})[configs[i].id] = cost
        experiments_used += which_exe.size

        not_executed_elite = which_alive[(is_elite[which_alive] > 0) & ~np.isin(which_alive, which_exe)]
        is_elite[not_executed_elite] -= 1

        # race.R:887-905: Inf denotes immediate rejection of a configuration.
        rejected_now = which_exe[np.isinf(Results[task, which_exe])]
        if rejected_now.size:
            if verbose:
                irace_note("Immediately rejected configurations: ",
                           ", ".join(str(configs[i].id) for i in rejected_now), "\n")
            is_rejected[rejected_now] = True
            is_elite[rejected_now] = 0
            alive[rejected_now] = False
            if not alive.any():
                raise RuntimeError("All configurations have been immediately rejected "
                                   "(all of them returned Inf)!")
            which_alive = np.where(alive)[0]
            nb_alive = len(which_alive)

        prev_nb_alive = nb_alive
        prev_which_alive = which_alive
        test_done = test_dropped = False
        if (task + 1) >= first_test and (task + 1) % scenario.each_test == 0 and nb_alive > 1:
            res = eliminate(Results[:task + 1, :], alive, which_alive, scenario.confidence)
            alive = res.alive
            test_dropped = nb_alive > int(alive.sum())
            test_done = True
        # Elites cannot be truly dropped while they still have unconsumed "safe"
        # historical instances pending (irace.R:941-945).
        tested_alive = alive.copy()
        alive = alive | (is_elite > 0)
        new_nb_alive = int(alive.sum())

        # race.R:951-955 (no capping).
        if test_dropped:
            res_symb = "-" if prev_nb_alive != new_nb_alive else "!"
        else:
            res_symb = "=" if test_done else "x"
        # race.R:961-970: the best among those alive before the test.
        best = prev_which_alive[0] if prev_which_alive.size == 1 else \
            prev_which_alive[np.argmin(get_ranks(Results[:task + 1, prev_which_alive]))]

        if elitist and (task + 1) > first_test and (task + 1) % scenario.each_test == 0 \
                and not np.any(is_elite > 0):
            no_elimination = no_elimination + 1 if new_nb_alive == prev_nb_alive else 0

        if verbose:
            race_print_task(res_symb, Results[:task + 1], inst_id, task + 1, np.where(alive)[0],
                            configs[best].id, best, experiments_used, start_time)
        if show_configurations:
            status = {i: "rejected" for i in rejected_now}
            status.update({i: "discarded" for i in prev_which_alive if not alive[i]})
            # The test dropped it, but it still has previous instances to see ("!").
            status.update({i: "kept (elite)" for i in prev_which_alive
                           if alive[i] and not tested_alive[i]})
            print_configurations(task, inst_id, task_alive, set(which_exe), status)

    if break_reason is None:
        break_reason = f"all instances ({len(race_instances)}) evaluated"

    # race.R:1016-1027: after rejections, a configuration that never got evaluated
    # cannot be ranked.
    if is_rejected.any():
        alive &= ~np.isnan(Results).all(axis=0)
    which_alive = np.where(alive)[0]
    if which_alive.size == 0:
        raise RuntimeError("All configurations have been eliminated; race produced no survivors.")

    ranks = overall_ranks(Results, which_alive)
    order = np.argsort(ranks, kind="stable")
    ranked = [configs[which_alive[o]] for o in order]
    for o in order:
        configs[which_alive[o]].rank = float(ranks[o])

    if verbose:
        # race.R:1030-1041: the final best and its mean over the instances any
        # configuration was evaluated on (NA if it missed one of them).
        final_best = which_alive[order[0]]
        used_rows = ~np.isnan(Results).all(axis=1)
        mean_best = float(np.mean(Results[used_rows, final_best]))
        old_best_id = None if best is None or best == final_best else configs[best].id
        race_print_footer(configs[final_best], mean_best, break_reason, debug_level,
                          old_best_id, scenario.parameters)

    return RaceResult(ranked_configs=ranked, experiments_used=experiments_used,
                       break_reason=break_reason, tasks_processed=tasks_processed,
                       rejected=[configs[i] for i in np.where(is_rejected)[0]])


def extract_elites(ranked_configs: list, n_elites: int) -> list:
    """Ports extractElites (irace.R:417-439): dedupe exact-duplicate parameter
    tuples, keep the best `n_elites`, and assign `.WEIGHT.`."""
    from sampling import elite_weight

    seen = set()
    unique = []
    for c in ranked_configs:
        k = c.key()
        if k in seen:
            continue
        seen.add(k)
        unique.append(c)

    elites = unique[:n_elites]
    n = len(elites)
    for rank, c in enumerate(elites, start=1):
        c.weight = elite_weight(n, rank)
    return elites
