import collections
import os
import re
import subprocess
from itertools import product

import numpy as np
import slugify


# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# SEEDS =  [93581]
WANDB = True
N_SEEDS = 5
MAX_PROCESSES = 2
base_name = "testnoise"

SEEDS =  [93581]
WANDB = False 

params = dict(
    base_match_sigma=2,
    match_sigma=2,
    base_internal_sigma=0.1,
    cum_match_stop_th=1.0,
)

# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------


def get_combinations(data):
    """
    Generates all possible combinations of list elements from a dictionary.

    Args:
       data: A dictionary.

    Yields:
       A dictionary representing a single combination of elements.
    """
    for k, v in data.items():
        if not isinstance(v, collections.abc.Iterable):
            data[k] = [v]

    combinations = product(*[value for value in data.values()])
    for combination in combinations:
        yield dict(zip(data.keys(), combination))


def optimize_option_key(options_str):
    """
    Generates an optimized option key from a string of options.

    Args:
        - options_str: A string containing options

    Returns:
        A slugified string representing the option key.
    """
    cleaned_str = options_str.replace("-o", "-").replace(" ", "")
    cleaned_str = re.sub(r"epochs=\d+", "", cleaned_str)
    return slugify.slugify(cleaned_str)


seeds = SEEDS or np.random.randint(0, 1e5, 5)
wandb = "-w" if WANDB else ""


processes = []

orig_path = os.path.dirname(os.path.realpath(__file__))

for i, p in enumerate(get_combinations(params)):
    for seed in seeds:
        # If MAX_PROCESSES reached, wait until all of them finish.
        if len(processes) == MAX_PROCESSES:
            for process in processes:
                process.wait()
            processes = []
        #
        options_str = ""
        for k, v in p.items():
            options_str += f" -o '{k}={v}'"
        option_key = optimize_option_key(options_str)

        base_cmd_str = (
            f"nohup python {orig_path}/SMMain.py "
            f"-n {base_name}_{option_key}_{seed:06d} "
            f"-s {seed} -t 55000 -x -g {wandb} "
            "--wdb_project grasp-simulation "
            "--wdb_entity francesco-mannella"
        )
        cmd_str = base_cmd_str + options_str

        print(f"Running: {cmd_str}")
        processes.append(subprocess.Popen(cmd_str, shell=True))

# wait for all processes
exit_codes = [p.wait() for p in processes]
