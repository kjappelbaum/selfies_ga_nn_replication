# -*- coding: utf-8 -*-
import subprocess
import time

import click

TEMPLATE = """#!/bin/bash
#SBATCH --chdir ./
#SBATCH --mem       32GB
#SBATCH --ntasks    1
#SBATCH --cpus-per-task   4
#SBATCH --time      18:00:00
#SBATCH --partition serial
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevin.jablonka@epfl.ch
#SBATCH --job-name={name}

source /home/kjablonk/anaconda3/bin/activate
conda activate ga_replication
python -u -m experiments.guacamol_baseline.benchmarks.goal_benchmark_rev_ga_original-- {beta} {watchtime} {num_generations}
"""

BETAS = [100, 1000]
NUM_GENERATIONS = [10, 100, 200, 400]
WATCHTIME = [5]


def write_script_and_submit(beta, num_generations, watchtime, submit):
    name = f"exp_guacamol_{beta}_{num_generations}"
    filled_script = TEMPLATE.format(
        **{
            "name": name,
            "beta": beta,
            "watchtime": watchtime,
            "num_generations": num_generations,
        }
    )

    with open(name + ".slurm", "w") as fh:
        fh.write(filled_script)

    if submit:
        subprocess.call("sbatch {}".format(name + ".slurm"), shell=True, cwd=".")
        time.sleep(10)


@click.command("cli")
@click.option("--submit", is_flag=True)
def main(submit):
    for beta in BETAS:
        for num_generation in NUM_GENERATIONS:
            if beta != 0:
                for watchtime in WATCHTIME:
                    write_script_and_submit(
                        beta, num_generation, watchtime, submit,
                    )


if __name__ == "__main__":
    main()
