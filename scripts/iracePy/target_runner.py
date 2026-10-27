"""Execution of the external target-runner CLI.

See MAPPING.md section G. Ports R_stuff/race-wrapper.R's run_target_runner and
target_runner_default. The command line is R's `targetCmdline` template, by default

    target_runner cand_id inst_id seed problem  --param1 v1 --param2 v2 ...

(the empty `{bound}` leaves two spaces), run through /bin/sh as R's system2 does.
Each parameter contributes its switch followed directly by its value, formatted as
R formats it (format_value). The default switch `--<name> ` gives `--name value`.

`inst_id` is the 1-based index of `problem` in the scenario's instance list, as in
R's `{instanceID}`; repeated copies of an instance share it, and `seed` tells them
apart.

The output must be whitespace-separated `cost [time]`. As in R, the output is stdout
and stderr together, so anything the runner writes to stderr (a warning, progress
text) counts as extra tokens and fails the checks below. A cost of `Inf` is valid and
rejects the configuration (see race.elitist_race). Anything else -- a non-zero exit
status, empty or non-numeric output such as `Error! ...`, more than two numbers, or a
time that is not a finite positive number -- aborts the whole run with TargetRunnerError, as R's target_error() does
(race-wrapper.R:61-89, 244-300, 479-505).
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess

import numpy as np

# The default targetCmdline (irace-options.R).
DEFAULT_TARGET_CMDLINE = "{configurationID} {instanceID} {seed} {instance} {bound} {targetRunnerArgs}"

# quote.param("targetRunner") in checkScenario (scenario.R:181-187), with Python's names.
_TEXT = "target runner 'target_runner' (--target-runner)"


def resolve_target_runner(path: str) -> str:
    """Ports checkScenario's target-runner handling (scenario.R:342-349): make the
    path absolute with path_rel2abs (path_rel2abs.R:16-39), then file.check(...,
    executable = TRUE) (utils.R:80-136). Raises ValueError with R's messages.

    As in R, a relative name that does not start with '.' is first looked up on
    PATH (Sys.which); if no executable is found there, it is taken relative to the
    working directory.
    """
    if not isinstance(path, str) or path == "":
        raise ValueError(f"{_TEXT} {path!r} is not a valid filename")
    path = os.path.expanduser(path)
    if not os.path.isabs(path) and not path.startswith("."):
        path = shutil.which(path) or path
    path = os.path.normpath(os.path.join(os.getcwd(), path))

    if not os.path.exists(path):
        raise ValueError(f"{_TEXT} '{path}' does not exist")
    if not os.access(path, os.R_OK):
        raise ValueError(f"{_TEXT} '{path}' is not readable")
    if not os.access(path, os.X_OK):
        raise ValueError(f"{_TEXT} '{path}' is not executable")
    if os.path.isdir(path):
        raise ValueError(f"{_TEXT} '{path}' is a directory, not a file")
    return path


def resolve_exec_dir(path: str) -> str:
    """Ports checkScenario's execDir handling (scenario.R:277-279): make the path
    absolute with path_rel2abs, then file.check(..., isdir = TRUE) (utils.R:80-136).
    Raises ValueError with R's messages."""
    text = "execution directory 'exec_dir' (--exec-dir)"
    if not isinstance(path, str) or path == "":
        raise ValueError(f"{text} {path!r} is not a valid filename")
    path = os.path.normpath(os.path.join(os.getcwd(), os.path.expanduser(path)))
    if not os.path.exists(path):
        raise ValueError(f"{text} '{path}' does not exist")
    if not os.path.isdir(path):
        raise ValueError(f"{text} '{path}' is not a directory")
    return path


class TargetRunnerError(RuntimeError):
    """A target-runner call failed; like R's irace_error, this stops the run."""


def _fail(err_msg: str, call: str, output: str, exec_dir: str | None = None) -> None:
    # Message layout follows target_error (race-wrapper.R:61-89).
    where = f" from the execution directory '{exec_dir}'" if exec_dir else ""
    raise TargetRunnerError(
        f"{err_msg}\nThe call to targetRunner was:\n{call}\nThe output was:\n{output}\n"
        "This is not a bug in irace, but means that something failed when running the "
        f"command above or it was terminated before completion. Try to run the command "
        f"above{where} to investigate the issue.")


def sh_quote(string: str) -> str:
    """Ports R's shQuote(type = "sh"): single quotes, or double quotes with
    `"`, `$`, `` ` `` and `\\` escaped when the string contains a single quote."""
    if "'" not in string:
        return f"'{string}'"
    return '"' + re.sub(r'(["$`\\])', r"\\\1", string) + '"'


def format_value(value) -> str:
    """Ports format.default(value, digits = 15L, scientific = FALSE) as used by
    buildCommandLine (race-wrapper.R:34-42): at most 15 significant digits, trailing
    zeros dropped, never scientific notation. So 2.0 gives "2" and 1e-05 gives
    "0.00001"."""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Inf" if value > 0 else "-Inf"
    if value == 0:
        return "0"
    # R's formatReal: the fewest significant digits (at most 15) that give the
    # same value as 15 do, then fixed notation with that many.
    mantissa, exponent = f"{value:.14e}".split("e")
    n_sig = len(mantissa.replace("-", "").replace(".", "").rstrip("0"))
    decimals = max(0, n_sig - 1 - int(exponent))
    return f"{value:.{decimals}f}"


def build_command_line(params: dict, switches: dict | None = None) -> str:
    """Ports buildCommandLine (race-wrapper.R:34-42): each switch is pasted directly
    in front of its value, and the pairs are joined by a space. The default switch
    is `--<name> `, i.e. two words."""
    switches = switches or {}
    return " ".join(switches.get(name, f"--{name} ") + format_value(value)
                    for name, value in params.items())


def check_target_cmdline(target_cmdline: str) -> None:
    """Ports check_target_cmdline (race-wrapper.R:378-386) without capping."""
    for x in ("seed", "instance", "targetRunnerArgs"):
        if "{" + x + "}" not in target_cmdline:
            raise ValueError(f"targetCmdline '{target_cmdline}' must contain '{{{x}}}'")


def expand_target_cmdline(target_cmdline: str, cand_id, inst_id, seed, problem,
                          target_runner: str, target_runner_args: str) -> str:
    """Ports expand_target_cmdline (race-wrapper.R:388-404): replace each variable
    in R's order. There is no capping, so `{bound}` is empty."""
    variables = {"configurationID": cand_id, "instanceID": inst_id, "seed": seed,
                 "instance": problem, "bound": "", "targetRunner": sh_quote(target_runner),
                 "targetRunnerArgs": target_runner_args}
    for name, value in variables.items():
        target_cmdline = target_cmdline.replace("{" + name + "}", str(value))
    return target_cmdline


def run_experiment(target_runner: str, cand_id: int, inst_id: int, seed: int, problem: str,
                    params: dict, *, switches: dict | None = None,
                    target_cmdline: str = DEFAULT_TARGET_CMDLINE,
                    exec_dir: str | None = None) -> tuple:
    """Ports run_target_runner and target_runner_default (race-wrapper.R:406-505).

    The runner runs in `exec_dir` (R's execDir, race-wrapper.R:515-518), or in the
    current directory if it is None. It is passed as the subprocess's working
    directory rather than by changing directory, so parallel workers are safe."""
    args = expand_target_cmdline(target_cmdline, cand_id, inst_id, seed, problem, target_runner,
                                 build_command_line(params, switches))
    # The call as R reports it: the runner unquoted, then the arguments.
    call = f"{target_runner} {args}"

    def fail(err_msg: str, output: str) -> None:
        _fail(err_msg, call, output, exec_dir)

    # runcommand (utils.R:416-454) calls system2(targetRunner, args, stdout = TRUE,
    # stderr = TRUE), which runs `shQuote(targetRunner) args 2>&1` through /bin/sh.
    # So the arguments are word-split and expanded by the shell, and stderr is
    # parsed together with stdout, in the order the runner wrote them.
    command = f"{sh_quote(target_runner)} {args} 2>&1"
    try:
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, cwd=exec_dir)
    except OSError as e:
        fail(f"Running targetRunner failed: {e}", "")
    output = result.stdout
    if result.returncode != 0:
        fail(f"targetRunner returned a non-zero exit status ({result.returncode})!", output)

    # parse_output (race-wrapper.R:46-59): every token becomes a number, NaN if it
    # is not one (as.numeric); then target_runner_default (race-wrapper.R:479-505)
    # and check_output_target_runner (race-wrapper.R:244-293), in R's order.
    values = [_as_numeric(t) for t in output.split()]
    if len(values) > 2:
        fail("The output of targetRunner should not be more than two numbers!", output)
    if not values:
        fail("The output of targetRunner must be one number 'cost'!", output)
    cost = values[0]
    if math.isnan(cost):
        fail("The cost returned by targetRunner is not numeric!", output)
    time_used = values[1] if len(values) > 1 else None
    if time_used is not None:
        if math.isnan(time_used):
            fail("The time returned by targetRunner is not numeric!", output)
        if math.isinf(time_used):
            fail("The time returned by targetRunner is not finite!", output)
        if time_used <= 0:
            fail(f"The value of time ({time_used}) returned by targetRunner must be strictly "
                 "positive!", output)
    return cost, time_used


def _as_numeric(token: str) -> float:
    try:
        return float(token)
    except ValueError:
        return math.nan
