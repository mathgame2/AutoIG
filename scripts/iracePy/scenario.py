"""Core data structures: parameters, scenario settings, and configurations.

See MAPPING.md section A for how these correspond to R_stuff/irace-options.R
(scenario defaults) and R_stuff/parameters.R (parameter definitions, numeric-only
subset).
"""

from __future__ import annotations

import math
import numbers
import warnings
from dataclasses import dataclass, field
from typing import Callable

from target_runner import (DEFAULT_TARGET_CMDLINE, check_target_cmdline, resolve_exec_dir,
                           resolve_target_runner)

# .irace_tolerance (aaa.R:26)
IRACE_TOLERANCE = math.sqrt(2.220446049250313e-16)


def valid_real_bound(x: float, digits: int) -> bool:
    """Ports valid_real_bound (parameters.R:153-158): `x` is representable with
    `digits` decimal places."""
    if x == 0:
        return True
    rx = round(x, digits)
    return abs(rx - x) <= IRACE_TOLERANCE * max(1.0, abs(x)) and digits >= -math.log10(abs(x))


def _is_whole(x) -> bool:
    return (isinstance(x, numbers.Real) and not isinstance(x, bool) and math.isfinite(x)
            and x == math.floor(x))


# Python's options by R type (irace-options.R): b(oolean), r(eal in [0, 1]) and
# i(nteger, non-negative). max_configurations is not an R option.
BOOL_OPTIONS = ("sample_instances", "soft_restart", "postselection", "elitist")
REAL_OPTIONS = ("confidence", "soft_restart_threshold")
INT_OPTIONS = ("seed", "first_test", "each_test", "mu", "elitist_new_instances", "elitist_limit",
               "nb_iterations", "max_experiments", "parallel", "debug_level")


@dataclass
class Parameter:
    """A single tunable numeric parameter.

    kind: "real" or "int". No categorical/ordinal/dependent parameters (out of
    scope, see plan). `digits` is the number of decimal places real values are
    rounded to (readParameters' default of 4, readParameters.R:85); it is ignored
    for "int".

    Validation ports Parameter() (parameters.R:168-211): integer bounds must be
    whole numbers, real bounds must be representable within `digits`, and
    lower < upper strictly -- R has no fixed numeric parameters.
    """

    name: str
    kind: str  # "real" | "int"
    lower: float
    upper: float
    digits: int = 4
    # The switch column of R's parameter file: pasted directly in front of the
    # value on the command line (buildCommandLine, race-wrapper.R:34-42), e.g.
    # "--alpha " gives "--alpha 0.5" and "--alpha=" gives "--alpha=0.5".
    switch: str | None = None

    def __post_init__(self) -> None:
        # readParameters (readParameters.R:177-181).
        if not _is_whole(self.digits) or not 1 <= self.digits <= 15:
            raise ValueError("'digits' must be an integer within [1, 15]")
        self.digits = int(self.digits)
        if self.switch is None:
            self.switch = f"--{self.name} "
        if self.kind not in ("real", "int"):
            raise ValueError(f"Unsupported parameter kind: {self.kind!r}")
        if self.kind == "int":
            if self.lower != math.floor(self.lower) or self.upper != math.floor(self.upper):
                raise ValueError(f"Parameter {self.name!r} of kind 'int': bounds must be integers")
            self.lower, self.upper = int(self.lower), int(self.upper)
        elif self.digits < 15 and not (valid_real_bound(self.lower, self.digits)
                                       and valid_real_bound(self.upper, self.digits)):
            raise ValueError(
                f"Domain bounds ({self.lower}, {self.upper}) of parameter {self.name!r} of kind "
                f"'real' must be representable within the given digits={self.digits}")
        if self.lower >= self.upper:
            raise ValueError(f"Parameter {self.name!r}: lower bound must be smaller than upper bound")


@dataclass
class Configuration:
    """A candidate configuration (one row of R's `configurations` data frame)."""

    id: int
    values: dict
    parent_id: int | None = None
    weight: float = 0.0
    # R's .RANK.: the rank in the last race that evaluated it (race.R:1050).
    rank: float | None = None

    def key(self):
        """Value tuple used to detect duplicate configurations (extractElites)."""
        return tuple(self.values[name] for name in sorted(self.values))


@dataclass
class Scenario:
    """Tuning scenario settings. Field names/defaults mirror R_stuff/irace-options.R
    for the subset of options this implementation supports."""

    parameters: list
    instances: list
    target_runner: str
    max_experiments: int

    seed: int | None = None
    first_test: int = 5
    each_test: int = 1
    mu: int = 5
    confidence: float = 0.95
    elitist_new_instances: int = 1
    elitist_limit: int = 2
    sample_instances: bool = True
    soft_restart: bool = True
    soft_restart_threshold: float = 1e-4
    max_configurations: int = 1024  # computeNbConfigurations default (irace.R:211)
    nb_iterations: int = 0  # 0 = auto (computeNbIterations), extended while budget remains
    parallel: int = 0  # experiments run at once within a race task; 0 or 1 = serial
    postselection: bool = True  # post-selection race on the leftover budget (psRace.R)
    elitist: bool = True  # False: each race uses only new instances and reuses no results
    # The target runner's command line (targetCmdline, race-wrapper.R:388-404).
    target_cmdline: str = DEFAULT_TARGET_CMDLINE
    # repairConfiguration (generation.R:18-28): called as repair(values, parameters)
    # on every newly sampled configuration; returns the {name: value} dict to use.
    repair_configuration: Callable | None = None
    # execDir: the directory the target runner is run in (race-wrapper.R:515-518).
    exec_dir: str = "./"
    # debugLevel: 0 prints R's default output, 1 adds its debugLevel >= 1 messages,
    # 2 (not as in R) also lists the configurations after each race task.
    debug_level: int = 0

    def __post_init__(self) -> None:
        """Ports the checks of checkScenario (scenario.R:222-263, 335-362, 431-453)
        and readInstances (scenario.R:694-695, 716-718) that apply to the supported
        options (blockSize is fixed at 1). Values are converted to the option's type,
        as R does."""
        # scenario.R:222-253: boolean, real [0, 1] and non-negative integer options.
        for name in BOOL_OPTIONS:
            value = getattr(self, name)
            if not isinstance(value, numbers.Real) or value not in (0, 1):
                raise ValueError(f"'{name}' must be either 0 or 1.")
            setattr(self, name, bool(value))
        for name in REAL_OPTIONS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Real) \
                    or not 0.0 <= value <= 1.0:  # also rejects NaN
                raise ValueError(f"'{name}' must be a real value within [0, 1].")
            setattr(self, name, float(value))
        for name in INT_OPTIONS:
            value = getattr(self, name)
            if name == "seed" and value is None:
                continue  # R's NA: a seed is drawn when the run starts
            if not _is_whole(value) or value < 0:
                raise ValueError(f"'{name}' must be a non-negative integer.")
            setattr(self, name, int(value))
        for name in ("first_test", "each_test"):
            if getattr(self, name) <= 0:
                raise ValueError(f"'{name}' must be positive")
        # scenario.R:312-317.
        if self.repair_configuration is not None and not callable(self.repair_configuration):
            raise ValueError("'repair_configuration' must be a function")
        self.target_runner = resolve_target_runner(self.target_runner)
        self.exec_dir = resolve_exec_dir(self.exec_dir)
        check_target_cmdline(self.target_cmdline)
        if not self.instances:
            # readInstances never returns an empty list.
            raise ValueError("No instances provided")
        if self.first_test % self.each_test != 0:
            raise ValueError("'first_test' must be a multiple of 'each_test'.")
        if not self.elitist and self.postselection:
            warnings.warn("The postselection race requires elitist racing, so it will be disabled.")
            self.postselection = False
        if self.mu < self.first_test:
            warnings.warn("Assuming 'mu = first_test' because 'mu' cannot be smaller.")
            self.mu = self.first_test
        # scenario.R:451-453 (maxTime is not supported; negatives failed above).
        if self.max_experiments == 0:
            raise ValueError("Tuning budget was not provided. Set 'max_experiments'.")
