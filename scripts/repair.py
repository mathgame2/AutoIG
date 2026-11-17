# repair function for generator configurations, passed to irace by run_tuning.py
# python port of repairConfiguration in scenario.R (used with R irace)
# if detailed-output/repair.eprime exists, every configuration sampled by irace is repaired by solving
# the repair model with conjure, so that it satisfies the generator's constraints between parameters.
# Without a repair model, configurations are returned unchanged.
# Paths are relative to runDir, which is the working directory of run_tuning.py.

import hashlib
import json
import os
import re

from utils import log, run_cmd

outputDir = "./detailed-output"


def repair_configuration(values, parameters):
    """values: {name: value} of a configuration sampled by irace; parameters: irace's Parameter list.
    Returns the repaired {name: value}, with exactly the same names."""

    # if there is no repairing model, just return the current configuration
    if not os.path.isfile(os.path.join(outputDir, "repair.eprime")):
        return values

    # unique hash of the configuration, used as id for all generated files
    # (unlike scenario.R, irace's configuration id is not part of the hash, so identical configurations share a cache entry)
    fileHash = hashlib.md5(
        " ".join(f"{name}={values[name]}" for name in sorted(values)).encode()
    ).hexdigest()
    baseFileName = fileHash

    # check if repairing results are already available
    outFile = f"{outputDir}/repairout-{baseFileName}.json"
    if os.path.isfile(outFile):
        with open(outFile) as f:
            return json.load(f)

    # make sure min<=max in configuration
    for param in sorted(p.name for p in parameters):
        maxParam = None
        if param.endswith("_min"):
            maxParam = param[:-3] + "max"
        elif param.endswith("Min"):
            maxParam = param[:-3] + "Max"
        if maxParam in values:
            values[param], values[maxParam] = sorted((values[param], values[maxParam]))

    # write down essence param file
    paramFile = f"{outputDir}/repair-{baseFileName}.param"
    with open(paramFile, "wt") as f:
        f.writelines(f"letting {name} be {val}\n" for name, val in values.items())

    # solve, using a seed derived from the hash
    seed = int(fileHash[:7], 16)
    cmd = (
        f"conjure solve repair.essence {paramFile} -o {outputDir} --use-existing-model repair.eprime"
        f' --solver-options "-timelimit 120 -varorder domoverwdeg -randomiseorder -randomseed {seed}"'
        " --copy-solutions=off"
    )
    log(cmd)
    output, returnCode = run_cmd(cmd)

    # read results
    solutionFile = f"{outputDir}/repair-repair-{baseFileName}-solution000001.solution"
    solution = None
    if returnCode == 0 and os.path.isfile(solutionFile):
        with open(solutionFile) as f:
            solution = dict(
                re.findall(r"letting\s+(?:repaired_)?(\w+)\s+be\s+(-?\d+)", f.read())
            )

    # delete unneccessary files generated
    baseTempName = f"{outputDir}/repair-repair-{baseFileName}"
    for endName in [".eprime-info", ".eprime-infor", ".eprime-minion", ".eprime-param", ".stats.json",
                    "-solution000001.eprime-solution", "-solution000001.solution"]:
        if os.path.isfile(baseTempName + endName):
            os.remove(baseTempName + endName)
    os.remove(paramFile)

    if solution is None:
        # scenario.R returns NULL here, which stops irace; we keep the configuration instead
        # (the generator will then most likely be unsat and the configuration scored Inf)
        print(f"ERROR while repairing configuration {values}, using it unrepaired. Conjure output:\n{output}")
        return values

    # convert results to configuration
    # NOTE: only work for integer values; a parameter missing from the solution keeps its value
    repaired = {name: int(solution.get(name, val)) for name, val in values.items()}

    # save results in case the same configuration is sampled again
    with open(outFile, "wt") as f:
        json.dump(repaired, f)

    return repaired
