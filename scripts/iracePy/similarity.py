"""Soft-restart similarity detection (numeric parameters only).

See MAPPING.md section F. Ports the numeric branch of
R_stuff/irace.R:39-196 (numeric.configurations.equal / similarConfigurations),
with the categorical-string blocking machinery dropped entirely since this
implementation has no categorical parameters.
"""

from __future__ import annotations

from scenario import Configuration, Parameter


def similar_configuration_ids(configs: list, params: list, threshold: float) -> set:
    """Return the set of configuration ids that are numerically "too similar" to at
    least one other configuration, i.e. every parameter's normalised (by domain
    range) absolute difference is within `threshold`.

    Ports irace.R:172-180 (the "no categorical" branch of similarConfigurations),
    using numeric.configurations.equal's per-parameter normalised distance
    (irace.R:73-79).
    """
    similar: set = set()
    ranges = {p.name: (p.upper - p.lower) or 1.0 for p in params}
    n = len(configs)
    for i in range(n):
        for j in range(i + 1, n):
            d = 0.0
            for p in params:
                dx = abs(configs[i].values[p.name] - configs[j].values[p.name]) / ranges[p.name]
                if dx > d:
                    d = dx
                if d > threshold:
                    break
            if d <= threshold:
                similar.add(configs[i].id)
                similar.add(configs[j].id)
    return similar
