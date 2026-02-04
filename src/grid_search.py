import argparse
import collections
import json
import os
import re
import subprocess
import sys
from itertools import product

import slugify


os.environ["OPENBLAS_NUM_THREADS"] = "1"


def parse_arguments():
    parser = argparse.ArgumentParser("Executes a parameter grid search schedule")
    parser.add_argument(
        "-w",
        "--wandb",
        action="store_true",
        help="Enable WANDB",
    )
    parser.add_argument(
        "-p",
        "--max_processes",
        type=int,
        default=2,
        help="Max processes",
    )
    parser.add_argument(
        "-n",
        "--base_name",
        type=str,
        default="testnoise",
        help="Base name",
    )
    parser.add_argument(
        "-c",
        "--combs",
        required=True,
        type=str,
        help="JSON of combinations parameters",
    )
    return parser.parse_args()


args = parse_arguments()

with open(args.combs, "r") as f:
    params = json.load(f)


def get_combinations(data):
    for k, v in data.items():
        if not isinstance(v, collections.abc.Iterable):
            data[k] = [v]
    combinations = product(*[value for value in data.values()])
    for combination in combinations:
        yield dict(zip(data.keys(), combination))


def optimize_option_key(options_str):
    cleaned_str = options_str.replace("-o", "-").replace(" ", "")
    cleaned_str = re.sub(r"epochs=\d+", "", cleaned_str)
    return slugify.slugify(cleaned_str)


processes = []
orig_path = os.path.dirname(os.path.realpath(__file__))

for i, p in enumerate(get_combinations(params)):
    if len(processes) == args.max_processes:
        for process in processes:
            process.wait()
        processes = []
    options = []
    for k, v in p.items():
        if k != "seeds":
            options.append("-o")
            options.append(f"{k}={v}")
        else:
            seed = v
    options.append("-o")
    options.append(f"name='{args.base_name}'")

    option_key = optimize_option_key("".join(options))

    run_id = f"{args.base_name}_{option_key}_{seed:06d}"

    command = [
        sys.executable,
        f"{orig_path}/SMMain.py",
        "-n",
        f"{run_id}",
        "-s",
        f"{seed}",
        "-t",
        "55000",
        "-x",
        "-g",
        "--wdb_project",
        "grasp-simulation",
        "--wdb_entity",
        "francesco-mannella",
    ]

    if args.wandb:
        command.append("-w")
    command.extend(options)

    print(f"Running: {' '.join(command)}")

    if not os.path.exists(f"simulations/{run_id}"):

        with open(f"{run_id}.log", "w") as log:
            processes.append(
                subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                    text=True,
                )
            )
    else:
        print(f"{run_id} simulation present")

exit_codes = [p.wait() for p in processes]
