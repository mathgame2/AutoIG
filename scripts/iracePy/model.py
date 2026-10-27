"""The per-elite, per-parameter sampling model: init / update / restart.

See MAPPING.md section C. Ports R_stuff/model.R, numeric ("r"/"i") branches only;
the categorical branches (dec_factor/add_factor/prob_max normalisation) are
dropped since this implementation has no categorical parameters.

A Model maps: parameter name -> configuration id -> (std, last_value).
"""

from __future__ import annotations

from scenario import Parameter

Model = dict


def init_sd_numeric(p: Parameter) -> float:
    """Ports init_sd_numeric (model.R:6-16), numeric-only (no log-transform/dependent
    branch)."""
    return (p.upper - p.lower) * 0.5


def initialise_model(params: list, elites: list) -> Model:
    """Ports initialiseModel (model.R:26-58), numeric branch only."""
    model: Model = {}
    for p in params:
        sd = init_sd_numeric(p)
        model[p.name] = {e.id: (sd, e.values[p.name]) for e in elites}
    return model


def update_model(params: list, elites: list, old_model: Model, n_new: int) -> Model:
    """Ports updateModel (model.R:60-122), numeric branch (lines 111-117) only.

    An elite surviving from a previous race keeps shrinking its own std; a
    freshly-promoted elite (no entry yet under its own id) inherits its parent's
    std via `.PARENT.` before shrinking (model.R:92-96).
    """
    n_params = len(params)
    num_factor = n_new ** (-1.0 / n_params)
    new_model: Model = {p.name: {} for p in params}
    for p in params:
        pname = p.name
        old_col = old_model.get(pname, {})
        for e in elites:
            source_id = e.id if e.id in old_col else e.parent_id
            old_std, _old_value = old_col[source_id]
            new_model[pname][e.id] = (old_std * num_factor, e.values[pname])
    return new_model


def restart_model(model: Model, restart_ids, params: list, n_new: int) -> Model:
    """Ports restartModel (model.R:130-176), numeric branch (lines 160-173) only.

    `restart_ids` must already be resolved to model keys (i.e. elite ids, with any
    newly-sampled non-elite configuration redirected to its `.PARENT.` id, mirroring
    model.R:139-141).
    """
    n_params = len(params)
    back_factor = n_new ** (2.0 / n_params)
    second_factor = n_new ** (-1.0 / n_params)
    for p in params:
        cap = init_sd_numeric(p) * second_factor
        column = model[p.name]
        for cid in restart_ids:
            if cid not in column:
                continue
            std, value = column[cid]
            column[cid] = (min(std * back_factor, cap), value)
    return model
