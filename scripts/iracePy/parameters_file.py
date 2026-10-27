"""Reader for the numeric subset of R irace's parameter file (readParameters.R).

Each non-empty line, after removing comments from `#` to the end of the line, is

    name "switch" i|r (lower, upper)

e.g. conjure's `Ag "-Ag " i (1, 10)`. Categorical (c), ordinal (o) and logarithmic
(`i,log`, `(..., log)`) parameters, conditions (`| cond`), fixed values and the
`[global]` / `[forbidden]` sections are outside the numeric-only subset this port
supports, so they are rejected rather than ignored.
"""

from __future__ import annotations

import re

from scenario import Parameter

_LINE = re.compile(
    r'^(?P<name>[A-Za-z_.][\w.]*)\s+"(?P<switch>[^"]*)"\s+(?P<kind>[ir])\s*'
    r'\(\s*(?P<lower>[^,()]+?)\s*,\s*(?P<upper>[^,()]+?)\s*\)$')

KINDS = {"i": "int", "r": "real"}


def read_parameters(path: str, digits: int = 4) -> list:
    """Parameters read from the parameter file at `path`, in file order. `digits` is
    passed to every Parameter, as R's readParameters(digits = 4)."""
    parameters = []
    names = set()
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = re.sub(r"#.*$", "", line).strip()
            if not line:
                continue
            where = f"{path}:{n}"
            m = _LINE.match(line)
            if m is None:
                raise ValueError(f"{where}: unsupported parameter line {line!r} (expected "
                                 "'name \"switch\" i|r (lower, upper)'; categorical, ordinal, "
                                 "log-scale and conditional parameters are not supported)")
            name = m["name"]
            if name in names:
                raise ValueError(f"{where}: duplicated parameter name {name!r}")
            try:
                lower, upper = float(m["lower"]), float(m["upper"])
            except ValueError:
                raise ValueError(f"{where}: bounds of parameter {name!r} must be numbers, "
                                 f"got ({m['lower']}, {m['upper']})") from None
            try:
                parameters.append(Parameter(name, KINDS[m["kind"]], lower, upper,
                                            digits=digits, switch=m["switch"]))
            except ValueError as e:
                raise ValueError(f"{where}: {e}") from None
            names.add(name)
    if not parameters:
        raise ValueError(f"No parameters found in '{path}'")
    return parameters
