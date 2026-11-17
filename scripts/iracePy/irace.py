"""Top-level irace loop: budget/iteration bookkeeping, sampling orchestration,
soft-restart, and driving successive races.

See MAPPING.md section A. Ports the outer `repeat` loop of R_stuff/irace.R:889-1217
(irace_run) and the budget/count formulas at irace.R:199-298, matching
pseudocode/core.txt.
"""

from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from instances import InstancePool
from model import initialise_model, restart_model, update_model
from psrace import post_selection
from race import elitist_race, extract_elites
from report import Timer, irace_note, print_elapsed, print_elite_configurations, r_num
from sampling import sample_from_model, sample_sobol_configs
from scenario import Scenario
from similarity import similar_configuration_ids


def compute_nb_iterations(n_params: int) -> int:
    """Ports computeNbIterations (irace.R:200); pseudocode/core.txt:34."""
    return math.floor(2 + math.log2(n_params))


def compute_min_survival(n_params: int) -> int:
    """Ports computeTerminationOfRace (irace.R:225); pseudocode/race.txt:1."""
    return math.floor(2 + math.log2(n_params))


def compute_budget(remaining_budget: int, index_iteration: int, nb_iterations: int) -> int:
    """Ports computeComputationalBudget (irace.R:203-205); pseudocode/core.txt:21-22."""
    return math.floor(remaining_budget / (nb_iterations - index_iteration + 1))


def compute_nb_configurations(current_budget: int, index_iteration: int, mu: int, each_test: int,
                               n_elites: int = 0, n_old_instances: int = 0, new_instances: int = 0,
                               max_configurations: int = 1024) -> int:
    """Ports computeNbConfigurations (irace.R:208-221); pseudocode/core.txt:26-32.

    A single formula serves both the "first race" case (call with the elite/
    instance args at their defaults of 0) and later elitist races.
    """
    saved_budget = n_elites * n_old_instances
    denom = max(mu + each_test * min(5, index_iteration),
                round_to_next_multiple(n_old_instances + new_instances, each_test))
    return min(math.floor((current_budget + saved_budget) / denom), max_configurations)


def compute_nb_configurations_elitist(current_budget: int, index_iteration: int, mu: int,
                                      each_test: int, n_elites: int, n_old_instances: int,
                                      new_instances: int, min_survival: int,
                                      max_configurations: int = 1024) -> tuple[int, int]:
    """Ports irace.R:930-957: N_j for an elitist race after the first, with R's
    fallback when it is not larger than min_survival.

    Returns (nb_configurations, new_instances). The caller keeps the returned
    new_instances for the rest of the run, as R stores it in
    race_state$elitist_new_instances.
    """
    nb_configurations = compute_nb_configurations(
        current_budget, index_iteration, mu, each_test, n_elites=n_elites,
        n_old_instances=n_old_instances, new_instances=new_instances,
        max_configurations=max_configurations)
    # If we don't have enough budget, do not evaluate new instances.
    # TODO figure out what this section is
    if nb_configurations <= min_survival:
        new_instances = 0
        nb_configurations = compute_nb_configurations(
            current_budget, index_iteration, mu, each_test, n_elites=n_elites,
            n_old_instances=n_old_instances, new_instances=new_instances,
            max_configurations=max_configurations)
    # If still not enough budget, then try to do at least one test.
    if nb_configurations <= min_survival:
        nb_configurations = compute_nb_configurations(
            current_budget, 1, 1, each_test, n_elites=n_elites,
            n_old_instances=n_old_instances, new_instances=0,
            max_configurations=max_configurations)
    return nb_configurations, new_instances


def round_to_next_multiple(x: int, d: int) -> int:
    """Ports round_to_next_multiple (utils.R:139)."""
    return x + d - 1 - (x - 1) % d


def compute_minimum_budget(scenario: Scenario, min_survival: int, nb_iterations: int) -> int:
    """Ports computeMinimumBudget (irace.R:229-297): a budget large enough for
    min_survival + 1 configurations in each of nb_iterations races."""
    each_test = scenario.each_test
    t_new = scenario.elitist_new_instances
    minimum_budget = (min_survival + 1) * nb_iterations
    if not scenario.elitist:
        # T_i = mu + T_each * min(5, i), strictest for i >= 5 (irace.R:289-294).
        return minimum_budget * (scenario.mu + 5 * each_test)
    if nb_iterations > 5:
        return minimum_budget * (scenario.mu + each_test + (nb_iterations - 5) * t_new
                                 + 5 * max(each_test, t_new))
    return minimum_budget * (scenario.mu + each_test + (nb_iterations - 1) * max(each_test, t_new))


def top_up_instances(pool: InstancePool, next_instance: int, remaining_budget: int,
                     min_survival: int) -> None:
    """Ports irace.R:1145-1150: before each race, make sure there are at least
    ceiling(remaining_budget / min_survival) instances not yet seen."""
    needed = math.ceil(remaining_budget / min_survival)
    if pool.size - (next_instance - 1) < needed:
        pool.generate(needed)


def post_selection_applies(scenario: Scenario, remaining_budget: int) -> bool:
    """irace_finish's condition for the post-selection race (irace.R:669), with
    blockSize fixed at 1."""
    return scenario.postselection and remaining_budget // scenario.each_test > 1


def irace_run(scenario: Scenario, verbose: bool = True, on_experiment=None,
              on_iteration=None) -> list:
    """Run irace and return the final elite configurations, best first.

    Two optional callbacks (not in R) report the run as it happens, in place of
    R's irace.Rdata:
      - on_experiment(iteration, instance, configuration, seed, cost, time) for
        every experiment run (see race.elitist_race). The post-selection race
        reports its experiments with the iteration number after the last one.
      - on_iteration(iteration, elites) after each iteration's race, with that
        iteration's elites, best first (R's iraceResults$allElites).

    With scenario.parallel > 1, one thread pool is kept for the whole run and shut
    down at the end, like R's start_parallel/stop_parallel (race_state.R:163-180);
    the experiments of each race task run in it (race.execute_experiments).
    Threads suffice because each worker only waits on a target-runner subprocess.
    """
    if scenario.parallel <= 1:
        return _irace_run(scenario, verbose, None, on_experiment, on_iteration)
    with ThreadPoolExecutor(max_workers=scenario.parallel, thread_name_prefix="irace-exp") as executor:
        return _irace_run(scenario, verbose, executor, on_experiment, on_iteration)


def _irace_run(scenario: Scenario, verbose: bool, executor, on_experiment, on_iteration) -> list:
    timer = Timer()
    debug_level = scenario.debug_level
    params = scenario.parameters
    if not params:
        raise ValueError("at least one parameter is required")
    # race_state.R:63-67: without a seed, one is drawn from the current random
    # state, trunc(runif(1, 1, .Machine$integer.max)), and reported (irace.R:869),
    # so any run can be repeated by passing it. The seed then seeds every RNG the
    # run uses (Python's `random`, NumPy's global generator used by scipy's
    # truncnorm, and the instance pool's). The scenario itself is not changed.
    seed = scenario.seed
    if seed is None:
        seed = random.randrange(1, 2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    n_params = len(params)
    # irace.R:698-701: a user-given value is a hard limit; 0 means auto.
    nb_iterations = math.floor(scenario.nb_iterations or compute_nb_iterations(n_params))
    min_survival = compute_min_survival(n_params)

    # checkMinimumBudget (irace.R:300-318), maxExperiments case.
    minimum_budget = compute_minimum_budget(scenario, min_survival, nb_iterations)
    if scenario.max_experiments < minimum_budget:
        raise ValueError("Insufficient budget: With the current settings, irace will require a "
                         f"value of 'max_experiments' of at least '{minimum_budget}'.")

    pool = InstancePool(scenario, seed=seed)
    pool.generate(max(1, math.ceil(scenario.max_experiments / min_survival)))

    global_results: dict = {}
    # Keys of configurations rejected for returning Inf; R forbids them
    # (race.R:1042-1045) so they are never sampled again.
    rejected_keys: set = set()
    rejected_ids: set = set()
    # R's allConfigurations and iraceResults$allElites, for the post-selection race.
    all_configs: dict = {}
    all_elites: list = []
    remaining_budget = scenario.max_experiments
    experiments_used = 0
    # irace.R:695: reported by catInfo even before the first iteration sets them.
    current_budget = nb_configurations = 0
    index_iteration = 1
    next_id = 0
    next_instance = 1
    elites: list = []
    model = None
    first_race = True
    # R's race_state$elitist_new_instances (race_state.R:43): the fallback in
    # compute_nb_configurations_elitist may set it to 0 for the rest of the run.
    run_new_instances = scenario.elitist_new_instances

    def cat_info(*parts, block: bool = True) -> None:
        """Ports catInfo (irace.R:644-657): a note, and with `block` the state of
        the run."""
        if not verbose:
            return
        irace_note(*parts, "\n")
        if block:
            print(f"# Iteration: {index_iteration}\n"
                  f"# nbIterations: {nb_iterations}\n"
                  f"# experimentsUsed: {experiments_used}\n"
                  f"# timeUsed: 0\n"
                  f"# remainingBudget: {remaining_budget}\n"
                  f"# currentBudget: {current_budget}\n"
                  f"# number of elites: {len(elites)}\n"
                  f"# nbConfigurations: {nb_configurations}", flush=True)

    def debug_note(*parts) -> None:
        if verbose and debug_level >= 1:
            irace_note(*parts)

    # irace.R:860-885.
    cat_info("Initialization\n",
             (f"# Elitist race\n"
              f"# Elitist new instances: {scenario.elitist_new_instances}\n"
              f"# Elitist limit: {scenario.elitist_limit}\n") if scenario.elitist
             else "# Non-elitist race\n",
             f"# nbIterations: {nb_iterations}\n"
             f"# minNbSurvival: {min_survival}\n"
             f"# nbParameters: {n_params}\n"
             f"# seed: {seed}\n"
             f"# confidence level: {r_num(scenario.confidence)}\n"
             f"# budget: {remaining_budget}\n"
             f"# mu: {scenario.mu}\n"
             f"# deterministic: FALSE\n", block=False)

    while True:
        if remaining_budget <= 0:
            cat_info("Stopped because budget is exhausted")
            break
        if index_iteration > nb_iterations:
            # irace.R:914-922: an auto-computed N_iter only plans the budget split;
            # keep iterating (each extra one gets all remaining budget) until a
            # budget guard below stops the run.
            if scenario.nb_iterations == 0:
                nb_iterations = index_iteration
            else:
                if debug_level >= 1:
                    cat_info("Limit of iterations reached", block=False)
                break

        current_budget = compute_budget(remaining_budget, index_iteration, nb_iterations)

        if first_race or not scenario.elitist:
            # irace.R:959-966: no elites' results to save, so the elite and
            # instance counts are zero.
            nb_configurations = compute_nb_configurations(
                current_budget, index_iteration, scenario.mu, scenario.each_test,
                max_configurations=scenario.max_configurations)
        else:
            nb_configurations, run_new_instances = compute_nb_configurations_elitist(
                current_budget, index_iteration, scenario.mu, scenario.each_test,
                n_elites=len(elites), n_old_instances=next_instance - 1,
                new_instances=run_new_instances, min_survival=min_survival,
                max_configurations=scenario.max_configurations)
        elitist_new_instances = 0 if first_race else run_new_instances

        if nb_configurations <= min_survival:
            cat_info("Stopped because there is not enough budget left to race more than ",
                     f"the minimum ({min_survival}).\n",
                     "# You may either increase the budget or set 'minNbSurvival' to a lower value.")
            break
        if not first_race and nb_configurations <= len(elites):
            cat_info("Stopped because ",
                     "there is not enough budget left to race newly sampled configurations.")
            break
        # irace.R:1010-1024: every configuration must reach the first test; in elitist
        # racing the elites only run on the new instances.
        if scenario.elitist:
            first_test_cost = (nb_configurations - len(elites)) * scenario.mu + \
                len(elites) * min(elitist_new_instances, scenario.mu)
        else:
            first_test_cost = nb_configurations * scenario.mu
        if first_test_cost > current_budget:
            cat_info("Stopped because there is not enough budget left to race all "
                     "configurations up to the first test (or mu).")
            break

        # irace.R:1026-1034.
        cat_info(f"Iteration {index_iteration} of {nb_iterations}\n",
                 f"# experimentsUsed: {experiments_used}\n",
                 f"# remainingBudget: {remaining_budget}\n",
                 f"# currentBudget: {current_budget}\n",
                 f"# nbConfigurations: {nb_configurations}", block=False)

        if first_race:
            if debug_level >= 1:
                cat_info(f"Sample {nb_configurations} configurations from Sobol distribution",
                         block=False)
            # irace.R:1047: the first race is sampled from a Sobol sequence.
            race_configs, next_id = sample_sobol_configs(params, nb_configurations, next_id,
                                                         repair=scenario.repair_configuration)
            if len(race_configs) <= min_survival:
                # elitist_race's precondition (race.R:654-657): the space is too
                # small to hold more than min_survival distinct configurations.
                raise ValueError(f"Not enough configurations ({len(race_configs)}) for a race "
                                 f"(minSurvival={min_survival})")
            new_configs = race_configs
        else:
            nb_new = nb_configurations - len(elites)
            debug_note("Update model\n")
            model = update_model(params, elites, model, nb_new)
            debug_note(f"Sample {nb_new} configurations from model\n")
            new_configs, next_id = sample_from_model(params, elites, model, nb_new, next_id,
                                                     forbidden=rejected_keys,
                                                     repair=scenario.repair_configuration)
            race_configs = elites + new_configs

            if scenario.soft_restart:
                flagged = similar_configuration_ids(race_configs, params, scenario.soft_restart_threshold)
                if flagged:
                    elite_ids = {e.id for e in elites}
                    resolved = {c.id if c.id in elite_ids else c.parent_id
                                for c in race_configs if c.id in flagged}
                    debug_note("Soft restart: ", " ".join(str(i) for i in sorted(flagged)), " !\n")
                    model = restart_model(model, resolved, params, nb_new)
                    new_configs, next_id = sample_from_model(
                        params, elites, model, nb_new, next_id - nb_new, forbidden=rejected_keys,
                        repair=scenario.repair_configuration)
                    race_configs = elites + new_configs

        all_configs.update((c.id, c) for c in new_configs)
        top_up_instances(pool, next_instance, remaining_budget, min_survival)

        debug_note("Launch race\n")
        result = elitist_race(
            race_configs, scenario, pool, global_results,
            next_instance=next_instance, elitist_new_instances=elitist_new_instances,
            max_exp=current_budget, min_survival=min_survival, first_race=first_race,
            verbose=verbose, executor=executor, debug_level=debug_level,
            iteration=index_iteration, on_experiment=on_experiment)

        remaining_budget -= result.experiments_used
        experiments_used += result.experiments_used
        rejected_keys.update(c.key() for c in result.rejected)
        rejected_ids.update(c.id for c in result.rejected)
        n_elites_out = min(min_survival, len(result.ranked_configs))
        debug_note("Extracting elites\n")
        elites = extract_elites(result.ranked_configs, n_elites_out)
        all_elites.append([e.id for e in elites])
        if on_iteration is not None:
            on_iteration(index_iteration, elites)
        if verbose:
            print_elite_configurations(elites, params, debug_level)

        # irace.R:1144: next_instance = nrow(iraceResults$experiments) + 1, i.e.
        # one past every instance with any result, including "future" instances
        # a race reached after exhausting the old ones. Evaluated ids are always
        # a contiguous 1..k, so this is the count of instances in global_results.
        next_instance = 1 + len(global_results)

        if first_race:
            debug_note("Initialise model\n")
            model = initialise_model(params, elites)
            first_race = False

        debug_note(f"End of iteration {index_iteration}\n")
        index_iteration += 1

    # irace_finish (irace.R:659-663).
    if verbose:
        print_elapsed(timer)

    # irace_finish (irace.R:659-672): race the elites of all iterations on the
    # budget left, and return the winners instead.
    if all_elites and post_selection_applies(scenario, remaining_budget):
        outcome = post_selection(scenario, pool, global_results, all_configs, all_elites,
                                 rejected_ids, remaining_budget, min_survival, seed,
                                 executor=executor, verbose=verbose, timer=timer,
                                 iteration=index_iteration, on_experiment=on_experiment)
        if outcome is not None:
            elites = outcome[0]
    return elites
