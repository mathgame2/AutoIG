# User guide: irace in Python (`pyimplement/`)

This guide explains how to tune an algorithm's parameters with the Python
implementation of irace in this folder. It is self-contained: you do not need to
know the R package. If you do know it, section 11 lists what is different.

Contents:

1. [What it does](#1-what-it-does)
2. [Installation](#2-installation)
3. [Quick start](#3-quick-start)
4. [How irace works](#4-how-irace-works)
5. [Setting up a tuning scenario](#5-setting-up-a-tuning-scenario)
6. [Running irace](#6-running-irace)
7. [Reading the output](#7-reading-the-output)
8. [Worked example: ACOTSP](#8-worked-example-acotsp)
9. [Running on a Slurm cluster](#9-running-on-a-slurm-cluster)
10. [Troubleshooting](#10-troubleshooting)
11. [Differences from R irace](#11-differences-from-r-irace)
12. [Running the test suite](#12-running-the-test-suite)

---

## 1. What it does

irace (*iterated racing*) finds good parameter settings for an algorithm. You
supply three things:

- the **parameters** to tune and their ranges;
- a set of **training instances**, the problems your algorithm will be run on;
- a **target runner**: a program that runs your algorithm with one parameter
  setting (a *configuration*) on one instance and prints its cost.

irace then calls the target runner many times, up to a **budget** of runs that
you set. It discards configurations that are statistically worse, and samples
new configurations near the best ones. At the end it returns the best
configurations it found. Lower costs are better. To maximise a quantity, have
the target runner print its negative.

This implementation follows the R irace package (version 4.x) and the paper
López-Ibáñez et al., *The irace package: Iterated racing for automatic algorithm
configuration*, Operations Research Perspectives 3 (2016). It supports:

- integer and real parameters;
- elitist racing (the default) and non-elitist racing;
- the Friedman test, and the Wilcoxon test when two configurations are left;
- soft restarts, and the post-selection race on the leftover budget;
- a budget counted in runs of the target runner;
- an external target runner (any executable), run in parallel if you ask.

It does **not** support categorical, ordinal or conditional parameters,
forbidden-configuration expressions, time budgets or capping. Section 8 shows how
to work around the first three; section 11 has the full list.

## 2. Installation

You need Python 3.11 or later with NumPy and SciPy; pytest is needed only for the
tests. The pinned versions are in [requirements.txt](requirements.txt):

```sh
cd pyimplement
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

There is nothing else to install. The modules are imported by name (`scenario`,
`irace`, ...), so either run Python from inside `pyimplement/`, or add the folder
to `sys.path` (section 6.2).

> **On the St Andrews HPC cluster**: the `.venv` checked into this folder was
> built on another machine and fails with `GLIBC_2.38 not found`. Use the conda
> environment that already has the packages instead:
> `PY=/software/conda/tw74/conda/envs/autoig/bin/python`, then `$PY cli.py ...`.

## 3. Quick start

The [demo/](demo/) folder tunes a toy problem whose best values are known: a
bowl-shaped cost with its minimum at `x = 1.5`, `y = -2`, `n = 4`. The target
runner is [demo/target_runner.py](demo/target_runner.py) and the 20 instances are
listed in [demo/instances.txt](demo/instances.txt).

From the command line:

```sh
cd pyimplement
python3 cli.py --target-runner ./demo/target_runner.py \
    --instances demo/instances.txt \
    --param x real -5 5 --param y real -5 5 --param n int 1 10 \
    --budget 1500 --seed 1
```

This takes about a minute. The run ends with the best configurations, best first:

```
# Best configurations (first number is the configuration ID; listed from best to worst according to the sum of ranks):
         x       y n
259 1.4666 -1.9809 4
180 1.4792 -1.9480 4
246 1.4283 -1.9435 4
# Best configurations as commandlines (first number is the configuration ID; listed from best to worst according to the sum of ranks):
259 --x 1.4666 --y -1.9809 --n 4
180 --x 1.4792 --y -1.948 --n 4
246 --x 1.4283 --y -1.9435 --n 4
```

`python3 demo/run_demo.py` runs the same scenario from Python and checks that the
result is close to the optimum.

## 4. How irace works

You do not need this section to use irace, but it explains the output and how to
choose a budget. Below, *n* is the number of parameters.

**Iterations.** A run is a sequence of iterations, each ending with a *race*. irace
plans `N_iter = floor(2 + log2(n))` iterations; for example 3 for 3 parameters and
5 for 11. It splits the remaining budget evenly over the iterations still to come.
If budget is left after the planned iterations, it carries on with more
iterations until the budget runs out.

**Sampling.**
- The first iteration samples configurations evenly across the parameter space,
  using a scrambled Sobol sequence.
- Later iterations sample around the best configurations so far, the *elites*.
  Each new configuration picks an elite as its parent, with better elites more
  likely to be picked. Each parameter value is then drawn from a normal
  distribution, centred on the parent's value and cut off at the parameter's
  bounds.
- The spread of that distribution narrows every iteration, so the search
  gradually focuses.
- Real values are rounded to `digits` decimal places (default 4).

The number of configurations in a race depends on the iteration's budget. It is
about 33 for the first race of an 11-parameter scenario with a budget of 1000.

**Racing.** A race runs every alive configuration on one instance after another:

1. After `first_test` instances (default 5), and then after every `each_test`
   instance (default 1), irace runs a statistical test on the costs so far. This is
   the Friedman test, or a paired Wilcoxon test when only two configurations are
   alive. Any configuration significantly worse than the best at the `confidence`
   level (default 0.95) is discarded.
2. The race stops as soon as one of these holds:
   - at most `minSurvival = floor(2 + log2(n))` configurations are alive;
   - the iteration's budget cannot pay for the next test;
   - two tests in a row (`elitist_limit`) discarded nothing;
   - every instance has been used.
3. The survivors, at most `minSurvival` of them, become the elites of the next
   iteration.

**Elitism** (the default). Elites are never re-run on an instance they have
already seen; their earlier costs are reused. Each race starts on new instances
(`elitist_new_instances`, default 1), then replays the instances seen before, so
new configurations must prove themselves on the same instances as the elites. An
elite is not discarded before the new configurations have caught up with it.

**Instances and seeds.** irace builds a sequence from shuffled copies of your
instance list, and gives every entry its own random seed. Every configuration
run on the same entry gets the same seed, so differences in cost come from the
configurations rather than from luck.

**Soft restart.** If newly sampled configurations are almost identical to each
other or to an elite (`soft_restart_threshold`), the sampling distributions of
those elites are widened again, and the new configurations are sampled again.

**Post-selection.** When the main loop stops with budget left over, the elites of
every iteration are raced against each other on that budget. The winners of this
race are the final result.

**Minimum budget.** irace refuses to start if the budget cannot pay for
`N_iter` races of `minSurvival + 1` configurations. For 3 parameters that is 96
runs, and for 11 parameters 300. A useful run usually needs many times more: the
paper uses 5000 runs for 11 parameters.

## 5. Setting up a tuning scenario

### 5.1 Parameters

Each parameter is integer (`int`) or real (`real`) with a closed range
`[lower, upper]`:

```python
from scenario import Parameter

Parameter("alpha", "real", 0.0, 5.0)                   # rounded to 4 decimals
Parameter("ants", "int", 5, 100)
Parameter("rho", "real", 0.01, 1.0, digits=2)          # rounded to 2 decimals
Parameter("tabu", "int", 1, 50, switch="--tabu-length=")
```

Rules, checked when the parameter is created:
- `lower < upper`, strictly, so a parameter with a single value is not allowed.
  Pass fixed values to your algorithm inside the target runner instead.
- Integer bounds must be whole numbers.
- Real bounds must have no more than `digits` decimal places. With the default
  of 4, `5.00001` is rejected.
- `digits` is between 1 and 15.

**Switch.** The switch is the text pasted directly in front of the value on the
target runner's command line. The default is `--<name> `, with a trailing space,
which gives `--ants 25`. `switch="--tabu-length="` gives `--tabu-length=12`.

**Order.** Parameters are passed to the target runner in the order you list them.

On the command line: `--param NAME int|real LOWER UPPER` (repeat it for every
parameter), `--switch NAME=SWITCH`, and `--digits N`, which applies to all real
parameters.

**Constraints between parameters.** If some combinations are invalid (for example
`min_size <= max_size`), supply a *repair function*. It is called on every newly
sampled configuration, before it is used:

```python
def repair(values, parameters):
    if values["min_size"] > values["max_size"]:
        values["min_size"], values["max_size"] = values["max_size"], values["min_size"]
    return values          # must contain every parameter, and nothing else

Scenario(..., repair_configuration=repair)
```

On the command line: `--repair-configuration my_repair.py:repair`, or
`--repair-configuration some.module:repair`.

**Categorical, ordinal and conditional parameters** are not supported directly.
Encode them as described in section 8.2.

### 5.2 Training instances

An instance file has one instance per line. Everything after `#` is a comment, and
blank lines are ignored:

```
# training instances
/data/tsp/2000-1.tsp
/data/tsp/2000-2.tsp
```

Load it with `instances.read_instances(path)`, or pass it to `--instances`. In
Python, `Scenario.instances` is just a list of strings, so you can build it any
way you like.

Points to know:
- An instance does not have to be a file. It is just text handed to the target
  runner, such as a benchmark function's name.
- irace shuffles the list for each pass through it. To keep the file's order,
  use `--no-sample-instances` (`sample_instances=False`).
- The command line is run through `/bin/sh`, so an instance line containing
  spaces arrives as several arguments. For example, `100-1.tsp --time 1` passes
  the instance and an instance-specific option. Write your target runner to
  expect that, or avoid spaces.
- Paths are passed on unchanged. Relative paths are relative to the execution
  directory (section 5.3), so absolute paths are the safest.

### 5.3 The target runner

The target runner is any executable file. If it is a script, it needs a shebang
line such as `#!/usr/bin/env python3` and `chmod +x`. irace runs one experiment
per call:

```
target_runner <configID> <instanceID> <seed> <instance>  --x 0.5 --y -1.2 --n 4
```

| Argument | Meaning |
|---|---|
| `configID` | Configuration ID, unique within the run; starts at 0 |
| `instanceID` | 1-based line number of the instance in your instance list; repeated passes share it |
| `seed` | Random seed for this run; pass it to your algorithm so that runs are reproducible |
| `instance` | The instance text |
| the rest | One `switch value` pair per parameter, in the order you defined them |

The runner must:
- **print** the cost: one number, or two numbers `cost time` where `time > 0`;
- **exit with status 0**.

A cost of `Inf` is allowed: it means "this configuration failed". That
configuration is dropped at once and never sampled again. Any other problem stops
the whole run with a `TargetRunnerError` that shows the exact command and its
output: a non-zero exit status, no output, text that is not a number, or more
than two numbers.

> **Watch out:** standard error is read together with standard output, as in R.
> A warning your algorithm prints to stderr therefore counts as output, and the
> run stops with "should not be more than two numbers". Send the algorithm's
> output to files, as below, and print only the cost.

A minimal Python target runner:

```python
#!/usr/bin/env python3
import subprocess, sys

config_id, instance_id, seed, instance = sys.argv[1:5]
params = sys.argv[5:]                          # e.g. ['--alpha', '1.5', '--ants', '25']

result = subprocess.run(["/path/to/my_solver", "-i", instance, "--seed", seed, *params],
                        capture_output=True, text=True)
if result.returncode != 0:
    sys.exit(f"solver failed: {result.stderr}")  # exits with status 1, so irace stops
cost = float(result.stdout.split()[-1])        # adapt to your solver's output
print(cost)
```

Test it by hand before tuning:
`./my_runner 0 1 123 /data/tsp/2000-1.tsp --alpha 1.5 --ants 25`
It should print one number.

**Execution directory.** The runner runs in `exec_dir` (`--exec-dir`; default:
the directory you started irace from). Point it at an empty scratch directory if
your runner writes files. The directory must already exist. If several irace runs
can happen at once, give each one its own directory.

**Custom command line.** `--target-cmdline` changes the order and form of the
arguments. The default is
`{configurationID} {instanceID} {seed} {instance} {bound} {targetRunnerArgs}`,
where `{bound}` is always empty. The template must contain `{seed}`,
`{instance}` and `{targetRunnerArgs}`.

### 5.4 The budget

`max_experiments` (`--budget`) is the total number of target-runner calls,
including the post-selection race. The time a run takes is roughly
`budget × time per call / parallel workers`.

Guidelines:
- The minimum is checked at start-up (section 4). Use at least 10 times that.
- The paper tunes 11 parameters with 5000 runs.
- More training instances than races need is fine. irace reuses instances, with
  new seeds, when it runs out.

## 6. Running irace

### 6.1 From the command line: `cli.py`

```sh
python3 cli.py --target-runner RUNNER --instances FILE --budget N \
               --param NAME KIND LOWER UPPER [--param ...] [options]
```

| Option | `Scenario` field | R option | Default | Meaning |
|---|---|---|---|---|
| `--target-runner` | `target_runner` | `targetRunner` | required | The executable (section 5.3); a bare name is looked up on `PATH`, then in the current directory |
| `--instances` | `instances` (a list) | `trainInstancesFile` | required | Instance file (section 5.2) |
| `--budget` | `max_experiments` | `maxExperiments` | required | Total target-runner calls |
| `--param NAME KIND LO HI` | `parameters` | parameter file | required | One parameter (section 5.1); repeatable |
| `--switch NAME=SWITCH` | `Parameter.switch` | switch column | `--NAME ` | Text before the value |
| `--digits` | `Parameter.digits` | `digits` | 4 | Decimal places of real values |
| `--seed` | `seed` | `seed` | random | Seed for the whole run; the one used is printed |
| `--parallel` | `parallel` | `parallel` | 0 | Experiments run at the same time (0 or 1: one at a time) |
| `--exec-dir` | `exec_dir` | `execDir` | `./` | Directory the runner runs in |
| `--first-test` | `first_test` | `firstTest` | 5 | Instances before the first test |
| `--each-test` | `each_test` | `eachTest` | 1 | Instances between later tests |
| `--mu` | `mu` | `mu` | 5 | Affects how many configurations a race gets; raised to `first_test` if smaller |
| `--confidence` | `confidence` | `confidence` | 0.95 | Confidence level of the tests |
| `--nb-iterations` | `nb_iterations` | `nbIterations` | 0 | 0: automatic, extended while budget remains; otherwise a hard limit |
| `--elitist-new-instances` | `elitist_new_instances` | `elitistNewInstances` | 1 | New instances at the start of each race |
| `--elitist-limit` | `elitist_limit` | `elitistLimit` | 2 | Stop a race after this many tests without elimination (0 = no limit) |
| `--no-elitist` | `elitist=False` | `elitist = 0` | elitist | Non-elitist racing: each race uses new instances only; disables post-selection |
| `--no-sample-instances` | `sample_instances=False` | `sampleInstances = 0` | shuffle | Keep the instance file's order |
| `--no-soft-restart` | `soft_restart=False` | `softRestart = 0` | on | Turn off soft restarts |
| `--soft-restart-threshold` | `soft_restart_threshold` | `softRestartThreshold` | 1e-4 | How close counts as "almost identical" (section 4) |
| `--no-postselection` | `postselection=False` | `postselection = 0` | on | Skip the post-selection race |
| `--repair-configuration` | `repair_configuration` | `repairConfiguration` | none | `FILE.py:NAME` or `MODULE:NAME` (section 5.1) |
| `--target-cmdline` | `target_cmdline` | `targetCmdline` | see 5.3 | Argument template |
| `--debug-level` | `debug_level` | `debugLevel` | 0 | More output (section 7) |
| `--quiet` | `irace_run(verbose=False)` | `quiet` | off | Print nothing |
| (Python only) | `max_configurations` | none | 1024 | Most configurations in one race |

`python3 cli.py --help` prints the same list. Invalid values are rejected before
any experiment runs, with R's error messages.

### 6.2 From Python

Use Python when your instances or parameters are generated by code, when you need
a repair function, or when you want to use the result directly:

```python
import sys
sys.path.insert(0, "/path/to/pyimplement")     # the modules are imported by name

from instances import read_instances
from irace import irace_run
from report import print_final
from scenario import Parameter, Scenario

scenario = Scenario(
    parameters=[Parameter("x", "real", -5.0, 5.0),
                Parameter("y", "real", -5.0, 5.0),
                Parameter("n", "int", 1, 10)],
    instances=read_instances("/path/to/pyimplement/demo/instances.txt"),
    target_runner="/path/to/pyimplement/demo/target_runner.py",
    max_experiments=1500,
    seed=1,
    parallel=4,
)
elites = irace_run(scenario, verbose=True)    # verbose=False prints nothing
print_final(elites, scenario.parameters)      # the table printed by cli.py

best = elites[0]
print(best.id, best.values)                   # e.g. 259 {'x': 1.4666, 'y': -1.9809, 'n': 4}
```

`irace_run` returns the final elites, best first, as `Configuration` objects with
these fields:
- `id`;
- `values`, a `{name: value}` dict;
- `parent_id`, the elite it was sampled from (`None` for the first iteration);
- `rank`, its sum of ranks in its last race;
- `weight`, its probability of being chosen as a parent.

`Scenario(...)` checks every option when it is created and raises `ValueError` on
bad input. It also makes the target-runner and execution-directory paths absolute.

### 6.3 Parallel runs

`parallel=N` (`--parallel N`) runs up to *N* experiments at the same time on the
machine irace runs on. Notes:
- The experiments of one instance run together. Instances are still taken one at
  a time, because each test decides which configurations run next.
- A parallel run gives exactly the same results as a serial run with the same
  seed.
- There is no point in going above the number of configurations alive in a race.
  Keep *N* at or below your CPU count, especially if your algorithm limits itself
  by CPU time.
- To spread the work over several machines, run several independent irace runs
  instead, for example one Slurm job per seed. Multi-node (MPI) execution is not
  supported.

### 6.4 Reproducibility

With a fixed `seed`, a run is fully repeatable: the same configurations, the same
instance order and seeds, the same result. This holds even in parallel, as long
as your target runner is deterministic for a given seed. Without a seed, irace
draws one and prints it (`# seed: 687542627`), so you can repeat that run later.
The same seed does **not** give the same run as R irace, because the random number
generators differ.

## 7. Reading the output

The output follows R irace's format. Excerpts below come from an ACOTSP run.

**Header.** Printed once:

```
# 2026-10-06 16:23:45 BST: Initialization
# Elitist race
# Elitist new instances: 1
# Elitist limit: 2
# nbIterations: 5          <- planned iterations, N_iter
# minNbSurvival: 5         <- a race stops at this many survivors
# nbParameters: 11
# seed: 42                 <- reuse this to repeat the run
# confidence level: 0.95
# budget: 1000
# mu: 5
# deterministic: FALSE
```

**Each iteration.** First a summary:

```
# 2026-10-06 16:23:45 BST: Iteration 1 of 5
# experimentsUsed: 0
# remainingBudget: 1000
# currentBudget: 200       <- budget for this iteration's race
# nbConfigurations: 33     <- configurations in the race, elites included
```

Then the race table, one row per instance:

```
+-+-----------+-----------+-----------+----------------+-----------+--------+-----+----+------+
| |   Instance|      Alive|       Best|       Mean best| Exp so far|  W time|  rho|KenW|  Qvar|
+-+-----------+-----------+-----------+----------------+-----------+--------+-----+----+------+
|x|          1|         33|         11|     34431017.00|         33|00:00:07|   NA|  NA|    NA|
|x|          2|         33|         11|     34405635.00|         66|00:00:07|+0.94|0.97|0.0032|
|x|          3|         33|         11|     34396576.33|         99|00:00:07|+0.91|0.94|0.0019|
|x|          4|         33|         30|     34434104.75|        132|00:00:07|+0.90|0.93|0.0015|
|-|          5|          5|         11|     34433319.00|        165|00:00:07|+0.07|0.25|0.7414|
```

The first column is a marker:

| Marker | Meaning |
|---|---|
| `x` | No test on this instance (before `first_test`, or between tests) |
| `-` | Test done; some configurations were discarded |
| `=` | Test done; nothing discarded, so more instances are needed |
| `!` | Test done; elites looked worse but were kept by the elitist rules |
| `.` | Every alive configuration already had a result on this instance |

The header also lists `c` (capping) and `:`. As in R, neither is ever printed.

The other columns:

| Column | Meaning |
|---|---|
| `Instance` | Position in irace's instance sequence; not the `instanceID` passed to the runner |
| `Alive` | Configurations still in the race |
| `Best`, `Mean best` | Best configuration so far in this race, and its mean cost over this race's instances |
| `Exp so far` | Experiments used by this race |
| `W time` | Wall-clock time spent on this instance |
| `rho`, `KenW`, `Qvar` | How consistently the configurations rank across instances. Values near 1 (`rho`, `KenW`) and near 0 (`Qvar`) mean the instances agree with each other |

Each iteration ends with the best configuration of the race and the elite table:

```
Best-so-far configuration:          93    mean value:      34480472.31
...
# ...: Elite configurations (first number is the configuration ID; listed from best to worst according to the sum of ranks):
```

**End of the run.** A line reports CPU time (the target runners' time included)
and wall-clock time. The post-selection race follows, if it applies. `cli.py` then
prints the final tables shown in section 3.

**Debug levels.**
- `--debug-level 1` adds R's extra notes: why each race stopped (e.g.
  `# Stopped because tests without elimination (2) >= elitistLimit (2)`), the
  sampling steps, and the configurations' IDs, parents, ranks and weights in the
  tables.
- `--debug-level 2`, which R does not have, also lists after each instance every
  alive configuration with its cost, mean, rank sum and whether it was discarded.
  This is useful for understanding a race, but very long.

## 8. Worked example: ACOTSP

[examples/acotsp/](examples/acotsp/) is the tuning example used throughout the R
user guide and the paper: 11 parameters of the ACOTSP ant-colony solver for the
travelling salesman problem.

### 8.1 Running it

```sh
cd pyimplement/examples/acotsp
./setup.sh      # downloads and compiles ACOTSP, downloads 400 TSP instances
```

This needs `curl`, `unzip`, `make` and a C compiler. It creates:
- `ACOTSPQAP-master/acotsp`, the solver;
- `instances/`, with the paper's 200 training and 200 test instances listed in
  `training-2000.txt` and `testing-2000.txt`;
- `acotsp-arena/`, the execution directory.

Then either:

```sh
# The test: a 1000-run budget, 1 s per ACOTSP run. About 5-7 minutes with 16 CPUs.
cd pyimplement && python3 -m pytest tests/test_acotsp.py -s

# The paper's setting: 5000 runs of 20 s each. Hours, even in parallel.
python3 examples/acotsp/run_acotsp.py --budget 5000 --time 20 --parallel 32 --seed 1
```

The test checks two things:
- The first iteration matches R's own ACOTSP run: 5 iterations planned, a budget
  of 200 and 33 configurations.
- The tuned configuration beats ACOTSP's default on held-out test instances. In
  one run, the mean tour length was 34.98M tuned against 41.04M for the default.

Its environment variables are `ACOTSP_BUDGET`, `ACOTSP_TIME`, `ACOTSP_PARALLEL`
and `ACOTSP_DEBUG_LEVEL`.

### 8.2 Encoding parameters irace cannot express directly

ACOTSP's R parameter file uses three features this implementation lacks.
[acotsp_scenario.py](examples/acotsp/acotsp_scenario.py) shows a way around each
one, which you can reuse for your own algorithm:

| R feature | Workaround |
|---|---|
| **Categorical**, e.g. `algorithm (as, mmas, eas, ras, acs)` | An `int` parameter `0..4`. The target runner maps the index back, e.g. `--algorithm 2` becomes `--eas` |
| **Conditional**, e.g. `q0` only when `algorithm == acs` | Always sample it; the target runner leaves it off the algorithm's command line when it does not apply |
| **Forbidden**, e.g. `(alpha == 0) & (beta == 0)` | A repair function that moves the configuration out of the forbidden region (section 5.1), or a runner that prints `Inf` for it |

These workarounds behave slightly differently from R:
- Integer-coded categories are sampled near the parent's index, so neighbouring
  categories are more likely.
- An inactive conditional parameter still has a value, and that value counts when
  deciding whether two configurations are the same.

Neither stops the tuning from working. MAPPING.md section I has the details.

## 9. Running on a Slurm cluster

irace runs on one machine and uses `--parallel` worker processes there. On a
cluster, run it inside a job that asks for one node and *N* CPUs, and use
`--parallel N`. Do not run long tunings on a login node.

```bash
#!/bin/bash
#SBATCH --job-name=irace
#SBATCH --partition=small-short    # check `sinfo -s` for your cluster's partitions
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

PY=/software/conda/tw74/conda/envs/autoig/bin/python   # any Python with numpy and scipy
cd /path/to/pyimplement
mkdir -p arena-$SLURM_JOB_ID                           # one execution directory per job

$PY -u cli.py --target-runner /path/to/my_runner --instances /path/to/instances.txt \
    --param alpha real 0 5 --param ants int 5 100 \
    --budget 5000 --seed 1 \
    --parallel $SLURM_CPUS_PER_TASK --exec-dir arena-$SLURM_JOB_ID
```

Submit with `sbatch irace.sbatch`, check it with `squeue -u $USER`, and follow
the log with `tail -f irace-<jobid>.out`. Python's `-u` makes the log appear as
it is written instead of when the buffer fills.

- **ACOTSP**: [examples/acotsp/test_acotsp.sbatch](examples/acotsp/test_acotsp.sbatch)
  runs the ACOTSP test as a job. It writes its output to `pyimplement/run.txt`.
- **Several independent runs**, e.g. the paper's 30 runs: submit one job per seed,
  each with its own `--exec-dir`, or use a job array with `--seed $SLURM_ARRAY_TASK_ID`.
- **Interactive use**: `srun --partition=interactive --cpus-per-task=16 --time=01:00:00 --pty bash`.

## 10. Troubleshooting

| Message | Cause and fix |
|---|---|
| `Insufficient budget: ... at least 'N'` | The budget is below the minimum (section 4). Raise `--budget`, or use fewer parameters |
| `Tuning budget was not provided. Set 'max_experiments'.` | `--budget 0` |
| `target runner ... does not exist` / `is not executable` | Check the path; `chmod +x` the runner; scripts need a shebang line |
| `execution directory 'exec_dir' ... does not exist` | Create the directory first |
| `Domain bounds ... must be representable within the given digits=4` | A real bound has too many decimals; round it, or raise `--digits` |
| `Parameter 'x': lower bound must be smaller than upper bound` | Equal or swapped bounds. Fixed values belong in the target runner |
| `targetRunner returned a non-zero exit status` | The runner or your algorithm failed. Run the command shown in the error by hand, from the execution directory |
| `The output of targetRunner should not be more than two numbers!` | The runner printed extra text, often a warning on stderr (section 5.3) |
| `The cost returned by targetRunner is not numeric!` | The runner printed something that is not a number, such as `Error` |
| `Not enough configurations (k) for a race` | The parameter space is too small to hold enough distinct configurations, e.g. a few narrow integer ranges. Widen the ranges, or tune fewer parameters |
| `irace tried 100 times to sample ... not forbidden` | New samples keep matching configurations that returned `Inf`, which can happen with a few narrow integer ranges. Check why the runner returns `Inf` |
| `Stopped because there is not enough budget left to race more than the minimum` | Normal end of a run. The message suggests `minNbSurvival`, which this implementation does not have; raise the budget instead |
| `GLIBC_2.38 not found` | The bundled `.venv` does not run on this system; see section 2 |

If a run behaves unexpectedly, rerun it with the printed seed and
`--debug-level 1` or `2`.

## 11. Differences from R irace

The algorithm is the same as R irace's for the features both support; the
differences are in what you can configure and how. All behavioural differences
are listed in [ALL_DIFFERENCES.md](ALL_DIFFERENCES.md), and
[MAPPING.md](MAPPING.md) maps each R function to its Python counterpart.

**What you will notice in use:**
- There are **no `scenario.txt` or `parameters.txt` files**. The scenario is
  given as command-line options or as a Python `Scenario` object.
- **Configuration IDs start at 0**, not 1.
- **Results are not saved to `irace.Rdata`.** They are the printed output and the
  value `irace_run` returns. Redirect the output to a file if you need a record.
- **The same seed gives a different run than in R**, because the random number
  generators differ. Within Python, a seed always gives the same run.

**R features that are not available:**
- categorical, ordinal, conditional and dependent-domain parameters, and log-scale
  sampling (see section 8.2 for workarounds);
- forbidden-configuration expressions;
- initial configurations (`configurationsFile`);
- time budgets (`maxTime`), capping (`boundMax`), `targetEvaluator`,
  `targetRunnerRetries` and `targetRunnerLauncher`;
- `deterministic`, `blockSize`, test types other than Friedman, `minNbSurvival`,
  `nbConfigurations`, `nbExperimentsPerIteration`;
- testing the final configurations on test instances (do it yourself, as the
  ACOTSP test does);
- `instancesDir`, recovery from a saved run, MPI and batch-cluster modes.

## 12. Running the test suite

```sh
cd pyimplement
python3 -m pytest tests -m "not slow"      # 264 tests, about a minute
python3 -m pytest tests -m slow -s         # the real ACOTSP tests (run setup.sh first)
```

The slow tests are skipped automatically if ACOTSP has not been set up.
