"""Generation of the problem-instance / seed pool.

Ports the instance-list part of R_stuff/irace.R:321-351 (generateInstances) and
pseudocode/util_functions.txt's instanceFlow(): the pool grows by whole copies of the
user-supplied instance list, each copy independently shuffled when
`scenario.sample_instances` is set, and each occurrence gets a fresh random seed.

Two kinds of id:
  - pool id: the 1-indexed position in the pool (R's row of instances_log). The race
    loop and the persistent results store key on it.
  - instance id: the 1-indexed position of the instance in `scenario.instances` (R's
    instances_log$instanceID). This is what the target runner receives, as R's
    `{instanceID}` (race.R:52-55, 67-71); repeated copies of an instance share it.
"""

from __future__ import annotations

import math
import random
import re


def read_instances(path: str) -> list:
    """Ports readInstances (scenario.R:692-725) for an instance file: remove
    comments from `#` to the end of the line, then whitespace, then empty lines.

    R removes only leading whitespace and leaves trailing whitespace for the shell
    to drop when it splits the command line; Python passes arguments without a
    shell, so it strips both ends here instead.
    """
    with open(path) as f:
        lines = [re.sub(r"#.*$", "", line.rstrip("\r\n")).strip() for line in f]
    instances = [line for line in lines if line]
    if not instances:
        raise ValueError(f"No instances found in '{path}' (whitespace and comments "
                         "starting with '#' are ignored)")
    return instances


class InstancePool:
    def __init__(self, scenario, seed: int | None = None):
        """`seed` overrides `scenario.seed`; irace_run passes the seed it drew for an
        unseeded scenario."""
        self._instances = scenario.instances
        self._sample_instances = scenario.sample_instances
        self._rng = random.Random(scenario.seed if seed is None else seed)
        self.problems: list = []
        self.instance_ids: list = []
        self.seeds: list = []

    @property
    def size(self) -> int:
        return len(self.problems)

    def generate(self, n: int) -> None:
        """Ports generateInstances(update = TRUE) (irace.R:321-351): append
        ceiling(n / len(instances)) copies of the instance list."""
        base = self._instances
        for _ in range(math.ceil(n / len(base))):
            order = list(range(len(base)))
            if self._sample_instances:
                self._rng.shuffle(order)
            for idx in order:
                self.problems.append(base[idx])
                self.instance_ids.append(idx + 1)
                self.seeds.append(self._rng.randint(1, 2**31 - 1))

    def ensure(self, n: int) -> None:
        """Grow the pool (by whole copies) until it holds at least `n` instances."""
        if self.size < n:
            self.generate(n - self.size)

    def problem_for(self, pool_id: int):
        return self.problems[pool_id - 1]

    def instance_id_for(self, pool_id: int) -> int:
        """Index of the instance in `scenario.instances` (1-based), as passed to the
        target runner."""
        return self.instance_ids[pool_id - 1]

    def seed_for(self, pool_id: int) -> int:
        return self.seeds[pool_id - 1]
