#!/usr/bin/env python3
"""Thin CLI entry point: build a Scenario from the command line and run irace.

Example:
    python3 cli.py --target-runner ./demo/target_runner.py \\
        --instances demo/instances.txt \\
        --param x real -5 5 --param y real -5 5 --param n int 1 10 \\
        --budget 1500
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os

from instances import read_instances
from irace import irace_run
from report import print_final
from scenario import Parameter, Scenario
from target_runner import DEFAULT_TARGET_CMDLINE


def parse_args():
    parser = argparse.ArgumentParser(description="Run the core irace algorithm.")
    parser.add_argument("--target-runner", required=True, help="the executable target runner; a bare name is looked up on PATH, then in the working directory")
    parser.add_argument("--instances", required=True, help="path to a file listing one instance per line; '#' starts a comment")
    parser.add_argument("--budget", type=int, required=True, dest="max_experiments",
                         help="maximum number of target-runner experiments")
    parser.add_argument("--param", action="append", nargs=4, metavar=("NAME", "KIND", "LOWER", "UPPER"),
                         default=[], help="a numeric parameter: NAME real|int LOWER UPPER (repeatable)")
    parser.add_argument("--switch", action="append", default=[], metavar="NAME=SWITCH",
                        help="command-line switch of parameter NAME, pasted directly before its "
                             "value (default '--NAME '; e.g. --switch 'alpha=--alpha=') (repeatable)")
    parser.add_argument("--target-cmdline", default=DEFAULT_TARGET_CMDLINE,
                        help="command line given to the target runner (default: %(default)r)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--first-test", type=int, default=5)
    parser.add_argument("--each-test", type=int, default=1)
    parser.add_argument("--mu", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--elitist-new-instances", type=int, default=1)
    parser.add_argument("--elitist-limit", type=int, default=2)
    parser.add_argument("--no-sample-instances", action="store_false", dest="sample_instances")
    parser.add_argument("--no-soft-restart", action="store_false", dest="soft_restart")
    parser.add_argument("--soft-restart-threshold", type=float, default=1e-4)
    parser.add_argument("--nb-iterations", type=int, default=0,
                        help="maximum number of iterations (0 = auto, extended while budget remains)")
    parser.add_argument("--digits", type=int, default=4,
                        help="decimal places real parameter values are rounded to (as in R's readParameters)")
    parser.add_argument("--parallel", type=int, default=0,
                        help="number of experiments to run at once (0 or 1 = serial)")
    parser.add_argument("--no-postselection", action="store_false", dest="postselection",
                        help="do not race the elites of all iterations on the leftover budget")
    parser.add_argument("--no-elitist", action="store_false", dest="elitist",
                        help="non-elitist racing: every race uses new instances only")
    parser.add_argument("--repair-configuration", metavar="FILE.py:NAME|MODULE:NAME", default=None,
                        help="function called as NAME(values, parameters) on every new "
                             "configuration, returning the repaired {name: value} dict")
    parser.add_argument("--exec-dir", default="./",
                        help="directory the target runner is run in (default: %(default)r)")
    parser.add_argument("--debug-level", type=int, default=0, choices=(0, 1, 2),
                        help="1 adds R's debugLevel >= 1 messages to the output; 2 also lists, "
                             "after each instance of a race, every alive configuration with its "
                             "scores and whether it was discarded")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def parse_switches(items: list, names: list) -> dict:
    """`--switch NAME=SWITCH` values as {name: switch}. One token per option, so a
    switch starting with '-' is not taken for another option."""
    switches = {}
    for item in items:
        name, sep, switch = item.partition("=")
        if not sep:
            raise SystemExit(f"--switch {item!r}: expected NAME=SWITCH")
        if name not in names:
            raise SystemExit(f"--switch {item!r}: unknown parameter {name!r}")
        switches[name] = switch
    return switches


def load_function(spec: str):
    """Load the function named by `FILE.py:NAME` or `MODULE:NAME`, as R finds
    repairConfiguration by name (scenario.R:312-317)."""
    target, sep, name = spec.rpartition(":")
    if not sep or not target or not name:
        raise SystemExit(f"--repair-configuration {spec!r}: expected FILE.py:NAME or MODULE:NAME")
    try:
        if target.endswith(".py"):
            module_spec = importlib.util.spec_from_file_location(
                os.path.splitext(os.path.basename(target))[0], target)
            if module_spec is None or not os.path.isfile(target):
                raise FileNotFoundError(f"no such file: {target!r}")
            module = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(target)
    except (ImportError, OSError) as e:
        raise SystemExit(f"--repair-configuration {spec!r}: {e}") from e
    function = getattr(module, name, None)
    if not callable(function):
        raise SystemExit(f"--repair-configuration {spec!r}: {name!r} is not a function in {target!r}")
    return function


def main() -> None:
    args = parse_args()
    if not args.param:
        raise SystemExit("at least one --param is required")

    switches = parse_switches(args.switch, [name for name, *_ in args.param])
    parameters = [Parameter(name, kind, float(lower), float(upper), digits=args.digits,
                            switch=switches.get(name))
                  for name, kind, lower, upper in args.param]
    instances = read_instances(args.instances)

    scenario = Scenario(
        parameters=parameters,
        instances=instances,
        target_runner=args.target_runner,
        max_experiments=args.max_experiments,
        seed=args.seed,
        first_test=args.first_test,
        each_test=args.each_test,
        mu=args.mu,
        confidence=args.confidence,
        elitist_new_instances=args.elitist_new_instances,
        elitist_limit=args.elitist_limit,
        sample_instances=args.sample_instances,
        soft_restart=args.soft_restart,
        soft_restart_threshold=args.soft_restart_threshold,
        nb_iterations=args.nb_iterations,
        parallel=args.parallel,
        postselection=args.postselection,
        elitist=args.elitist,
        target_cmdline=args.target_cmdline,
        repair_configuration=(load_function(args.repair_configuration)
                              if args.repair_configuration else None),
        exec_dir=args.exec_dir,
        debug_level=args.debug_level,
    )

    elites = irace_run(scenario, verbose=not args.quiet)
    if not args.quiet:
        print_final(elites, parameters)


if __name__ == "__main__":
    main()
