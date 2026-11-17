#!/usr/bin/env python3

# run the instance generation tuning with the python implementation of irace (scripts/iracePy)
# must be called from runDir (run.sh does this), and uses the files created there by setup.py:
#   - config.json: all settings of the experiment
#   - params.json: the generator parameters to tune
#   - instances: irace's instance file (a single dummy instance)
# writes into runDir:
#   - irace-experiments.csv: every experiment run by irace (replaces experiment_log of R irace's irace.Rdata)
#   - irace-results.json: the elites of each iteration and the final elites

import csv
import json
import os
import sys

scriptDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(scriptDir, "iracePy"))

from instances import read_instances
from irace import irace_run
from report import print_final
from scenario import Parameter, Scenario

from repair import repair_configuration

experimentsFile = "irace-experiments.csv"
resultsFile = "irace-results.json"


class ExperimentLog:
    """Writes each experiment as soon as irace reports it, so a run that is killed still leaves its log."""

    columns = ["iteration", "instance", "configuration", "seed", "cost", "time"]

    def __init__(self, fileName):
        self.fileName = fileName

    def __enter__(self):
        self.file = open(self.fileName, "wt", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.columns)
        return self

    def __exit__(self, *exc):
        self.file.close()

    def write(self, iteration, instance, configuration, seed, cost, time):
        self.writer.writerow([iteration, instance, configuration, seed, cost, "" if time is None else time])
        self.file.flush()


def configuration_to_dict(configuration):
    return {"id": configuration.id, "values": configuration.values, "parent": configuration.parent_id}


def main():
    with open("config.json") as f:
        config = json.load(f)
    with open("params.json") as f:
        parameters = [Parameter(**p) for p in json.load(f)]

    scenario = Scenario(
        parameters=parameters,
        instances=read_instances("instances"),
        target_runner=os.path.join(scriptDir, "target-runner"),
        max_experiments=int(config["maxEvaluations"]),
        seed=int(config["seed"]),
        parallel=int(config["nCores"]),
        elitist=config["elite"] == "1",
        repair_configuration=repair_configuration,
        exec_dir="./",
        debug_level=int(config.get("debugLevel", 0)),
    )

    iterationElites = []

    def on_iteration(iteration, elites):
        iterationElites.append({"iteration": iteration, "elites": [e.id for e in elites]})

    with ExperimentLog(experimentsFile) as experimentLog:
        elites = irace_run(scenario, verbose=True, on_experiment=experimentLog.write,
                           on_iteration=on_iteration)

    with open(resultsFile, "wt") as f:
        json.dump(
            {
                "iterationElites": iterationElites,
                "finalElites": [configuration_to_dict(e) for e in elites],
            },
            f,
            indent=1,
        )

    print_final(elites, parameters)


main()
