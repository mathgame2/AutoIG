"""Console output in the format of R irace: timestamped notes, the race table printed
instance by instance, the race footer, and configuration tables.

Ports the printing helpers of R_stuff/race.R:299-412 (race table), R_stuff/utils.R:71
(irace_note), R_stuff/utils.R:350-410 (concordance, dataVariance),
R_stuff/configurations.R:105-145 (configurations_print, configurations_print_command)
and R_stuff/irace.R:620-663 (final report, elapsed time).
"""

from __future__ import annotations

import math
import numbers
import os
import time
from datetime import datetime

import numpy as np
from scipy.stats import rankdata

from target_runner import build_command_line

# .irace.format.perf (zzz.R:3).
FORMAT_PERF = "%#16.10g"
# test.type.order.str (utils.R:223-230) for the only test supported, friedman.
ORDER_STR = "sum of ranks"


def r_num(x) -> str:
    """A number as R's cat() prints it: at most 7 significant digits."""
    if isinstance(x, bool):
        return "TRUE" if x else "FALSE"
    if isinstance(x, numbers.Integral):
        return str(int(x))
    return format(float(x), ".7g")


def irace_note(*parts) -> None:
    """Ports irace_note (utils.R:71-78)."""
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"# {stamp}: " + "".join(str(p) for p in parts), end="", flush=True)


def format_perf(x: float) -> str:
    if x is None or math.isnan(x):
        return "%16s" % "NA"
    if math.isinf(x):
        return "%16s" % ("Inf" if x > 0 else "-Inf")
    return FORMAT_PERF % x


# ---------------------------------------------------------------------------
# Race table (race.R:299-412)

def table_hline(widths) -> str:
    return "+" + "+".join("-" * w for w in widths) + "+\n"


def table_sprintf(text, widths) -> str:
    return "|" + "|".join("%*s" % (w, t) for t, w in zip(text, widths)) + "|\n"


NOCAP_TABLE_FIELDS_WIDTH = (1, 11, 11, 11, 16, 11, 8, 5, 4, 6)
NOCAP_COLUMN_NAMES = (" ", "Instance", "Alive", "Best", "Mean best", "Exp so far",
                      "W time", "rho", "KenW", "Qvar")
NOCAP_HLINE = table_hline(NOCAP_TABLE_FIELDS_WIDTH)
RACE_COMMON_HEADER = """# Markers:
     x No test is performed.
     c Configurations are discarded only due to capping.
     - The test is performed and some configurations are discarded.
     = The test is performed but no configuration is discarded.
     ! The test is performed and configurations could be discarded but elite configurations are preserved.
     . Alive configurations were already evaluated on this instance and nothing is discarded.
     : All alive configurations are elite, but some need to be evaluated on this instance.

"""
NOCAP_HEADER = (RACE_COMMON_HEADER + NOCAP_HLINE
                + table_sprintf(NOCAP_COLUMN_NAMES, NOCAP_TABLE_FIELDS_WIDTH) + NOCAP_HLINE)


def race_print_header() -> None:
    print(NOCAP_HEADER, end="", flush=True)


def elapsed_wctime_str(now: float, start: float) -> str:
    """Ports elapsed_wctime_str (race.R:333-338)."""
    if now <= start:
        return "00:00:00"
    return time.strftime("%H:%M:%S", time.gmtime(now - start))


def concordance(data: np.ndarray) -> tuple[float, float]:
    """Ports concordance (utils.R:350-385): (Kendall's W, Spearman's rho) of a
    matrix with instances in rows and configurations in columns."""
    n, k = data.shape
    if n <= 1 or k <= 1:
        return math.nan, math.nan
    r = rankdata(data, axis=1, method="average")
    tie_counts = [np.unique(row, return_counts=True)[1] for row in r]
    # If everything is tied, then W = 1, perfect homogeneity.
    if all(len(c) == 1 for c in tie_counts):
        w = 1.0
    else:
        ties = float(sum(np.sum(c ** 3 - c) for c in tie_counts))
        R = r.sum(axis=0)
        w = (12 * np.sum((R - n * (k + 1) / 2) ** 2)) / ((n ** 2 * (k ** 3 - k)) - (n * ties))
    rho = (n * w - 1) / (n - 1)
    return float(w), float(rho)


def data_variance(data: np.ndarray) -> float:
    """Ports dataVariance (utils.R:391-410): mean variance of the configurations
    over the per-instance z-scores."""
    if data.shape[0] <= 1 or data.shape[1] <= 1:
        return math.nan
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, ddof=1, keepdims=True)
    std[std == 0] = 1
    return float(np.mean(((data - mean) / std).var(axis=0, ddof=1)))


def race_task_line(res_symb: str, results: np.ndarray, instance: int, current_task: int,
                   which_alive, id_best: int, best: int, experiments_used: int,
                   start_time: float, now: float | None = None) -> str:
    """Ports race_print_task_common (race.R:340-369). `results` holds the rows of
    the tasks seen so far (Results[seq_len(current_task), ]) and `current_task`
    is 1-based."""
    now = time.time() if now is None else now
    s = "|%s|%11d|" % (res_symb, instance)
    mean_best = float(np.mean(results[:, best]))
    s += ("%11d|%11d|" % (len(which_alive), id_best) + format_perf(mean_best)
          + "|%11d|%s" % (experiments_used, elapsed_wctime_str(now, start_time)))
    if current_task > 1 and len(which_alive) > 1:
        res = results[:, which_alive]
        kendall_w, spearman_rho = concordance(res)
        qvar = data_variance(res)
        s += "|%+4.2f|%.2f|%.4f|\n" % (spearman_rho, kendall_w, qvar)
    else:
        s += "|   NA|  NA|    NA|\n"
    return s


def race_print_task(*args, **kwargs) -> None:
    """Ports race_print_task_nocap (race.R:385-397)."""
    print(race_task_line(*args, **kwargs), end="", flush=True)


def race_print_configurations(instance: int, current_task: int, rows: list) -> None:
    """Not in R (debug_level >= 2): the configurations alive at the start of a
    task. `rows` holds (id, executed, cost, mean cost, rank sum or None, status);
    `executed` is False when the cost is a previous result for this instance.
    The rank sum is over the instances seen so far, among the configurations not
    rejected, as the elimination test ranks them."""
    out = [f"#   Configurations on instance {instance} (task {current_task}):",
           "#   %11s|%5s|%16s|%16s|%10s| %s" % ("ID", "Exec", "Cost", "Mean cost", "Rank sum",
                                                 "Status")]
    for cid, executed, cost, mean_cost, rank_sum, status in rows:
        out.append("#   %11d|%5s|%s|%s|%10s| %s" % (
            cid, "yes" if executed else "no", format_perf(cost), format_perf(mean_cost),
            "NA" if rank_sum is None else "%.1f" % rank_sum, status))
    print("\n".join(out), flush=True)


def race_print_footer(bestconf, mean_best: float, break_msg: str, debug_level: int,
                      old_best_id, parameters) -> None:
    """Ports race_print_footer (race.R:399-412)."""
    out = NOCAP_HLINE
    if debug_level >= 1:
        out += f"# Stopped because {break_msg}\n"
    if old_best_id is not None:
        out += f"Best configuration for the instances in this race: {old_best_id}\n"
    out += ("Best-so-far configuration: %11d" % bestconf.id + "    mean value: "
            + format_perf(mean_best) + "\n" + "Description of the best-so-far configuration:\n")
    print(out, end="")
    configurations_print([bestconf], parameters, metadata=True)
    print("", flush=True)


# ---------------------------------------------------------------------------
# Configuration tables (configurations.R:105-145)

def _is_na(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _format_real_column(values: list, digits: int = 15) -> list:
    """R's formatReal (format.c) with print(digits = 15) and scipen = 0: one common
    format for the whole column, fixed notation unless scientific is narrower."""
    finite = [float(v) for v in values if not _is_na(v)]
    if not finite:
        return ["NA"] * len(values)
    neg = any(v < 0 for v in finite)
    mxsl = mF = 1
    rgt = 0
    max_exp = 0
    for v in finite:
        if v == 0:
            kpower, nsig = 0, 1
        else:
            mantissa, exponent = f"{abs(v):.{digits - 1}e}".split("e")
            kpower = int(exponent)
            nsig = len(mantissa.replace(".", "").rstrip("0")) or 1
        mxsl = max(mxsl, kpower + 1 if kpower >= 0 else 1)
        rgt = max(rgt, nsig - kpower - 1)
        mF = max(mF, nsig)
        max_exp = max(max_exp, abs(kpower))
    fixed_width = neg + mxsl + (rgt + 1 if rgt > 0 else 0)
    sci_width = neg + (mF + 1 if mF > 1 else mF) + (5 if max_exp >= 100 else 4)
    if fixed_width <= sci_width:
        return ["NA" if _is_na(v) else f"{float(v):.{rgt}f}" for v in values]
    return ["NA" if _is_na(v) else f"{float(v):.{mF - 1}e}" for v in values]


def _format_column(values: list, integer: bool) -> list:
    if all(_is_na(v) for v in values):
        return ["NA"] * len(values)
    if integer or all(_is_na(v) or (isinstance(v, (numbers.Integral, np.integer))
                                    and not isinstance(v, bool)) for v in values):
        return ["NA" if _is_na(v) else str(int(v)) for v in values]
    if all(_is_na(v) or isinstance(v, numbers.Real) for v in values):
        return _format_real_column(values)
    return ["NA" if _is_na(v) else str(v) for v in values]


def configurations_print(configs: list, parameters: list, metadata: bool = False,
                         elite: bool = False) -> None:
    """Ports configurations_print (configurations.R:105-115), i.e.
    print.data.frame(digits = 15) with the configuration IDs as row names.

    With metadata the columns are .ID., the parameters and .PARENT., plus .RANK.
    and .WEIGHT. for `elite` configurations (the columns extractElites adds)."""
    kinds = {p.name: p.kind for p in parameters}
    names = [p.name for p in parameters]
    columns = []  # (header, formatted values)
    if metadata:
        columns.append((".ID.", _format_column([c.id for c in configs], True)))
    for name in names:
        columns.append((name, _format_column([c.values.get(name) for c in configs],
                                             kinds[name] == "int")))
    if metadata:
        columns.append((".PARENT.", _format_column([c.parent_id for c in configs], True)))
        if elite:
            columns.append((".RANK.", _format_column([c.rank for c in configs], False)))
            columns.append((".WEIGHT.", _format_column([c.weight for c in configs], False)))

    rownames = [str(c.id) for c in configs]
    rn_width = max(len(r) for r in rownames) if rownames else 0
    widths = [max(len(h), *(len(v) for v in vals)) if vals else len(h) for h, vals in columns]
    lines = [" " * rn_width + "".join(" %*s" % (w, h) for (h, _), w in zip(columns, widths))]
    for i, rn in enumerate(rownames):
        lines.append("%-*s" % (rn_width, rn)
                     + "".join(" %*s" % (w, vals[i]) for (_, vals), w in zip(columns, widths)))
    print("\n".join(lines), flush=True)


def configurations_print_command(configs: list, parameters: list) -> None:
    """Ports configurations_print_command (configurations.R:132-145)."""
    if not configs:
        return
    switches = {p.name: p.switch for p in parameters}
    width = len(str(max(c.id for c in configs)))
    for c in configs:
        values = {p.name: c.values[p.name] for p in parameters if not _is_na(c.values.get(p.name))}
        print("%-*d %s" % (width, c.id, build_command_line(values, switches)))


def print_elite_configurations(elites: list, parameters: list, debug_level: int) -> None:
    """irace.R:1193-1196 and psRace.R:266-270."""
    irace_note("Elite configurations (first number is the configuration ID;",
               " listed from best to worst according to the ", ORDER_STR, "):\n")
    configurations_print(elites, parameters, metadata=debug_level >= 1, elite=True)


def print_final(elites: list, parameters: list) -> None:
    """The report irace_common prints after the run (irace.R:620-628)."""
    print("# Best configurations (first number is the configuration ID;"
          f" listed from best to worst according to the {ORDER_STR}):")
    configurations_print(elites, parameters)
    print("# Best configurations as commandlines (first number is the configuration ID;"
          " same order as above):")
    configurations_print_command(elites, parameters)


class Timer:
    """Ports Timer (timer.R): CPU times include those of finished child processes
    (the target runners), as R adds proc.time()'s child times."""

    def __init__(self):
        self.start = os.times()

    def elapsed(self) -> tuple[float, float, float]:
        now = os.times()
        user = (now.user - self.start.user) + (now.children_user - self.start.children_user)
        system = (now.system - self.start.system) + (now.children_system - self.start.children_system)
        return user, system, now.elapsed - self.start.elapsed


def print_elapsed(timer: Timer) -> None:
    """irace.R:661-663."""
    user, system, wallclock = timer.elapsed()
    print(f"# Total CPU user time: {r_num(user)}, CPU sys time: {r_num(system)}, "
          f"Wall-clock time: {r_num(wallclock)}", flush=True)
