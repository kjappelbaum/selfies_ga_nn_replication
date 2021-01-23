# -*- coding: utf-8 -*-
import subprocess
import time

import click

TEMPLATE = """#!/bin/bash
#SBATCH --chdir ./
#SBATCH --mem       32GB
#SBATCH --ntasks    1
#SBATCH --cpus-per-task   4
#SBATCH --time      14:00:00
#SBATCH --partition serial
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kevin.jablonka@epfl.ch
#SBATCH --job-name={name}

source /home/kjablonk/anaconda3/bin/activate
conda activate ga_replication

python -u -m experiments.experiment_2ga.core_ga {beta} {iter}
"""

BETAS = [-100, -50, -10, -0.1, 0, 0.1, 10, 50, 100]
REPEATS = 5


@click.command("cli")
@click.option("--submit", is_flag=True)
def main(submit):
    for beta in BETAS:
        for repeat in range(REPEATS):
            name = f"exp_1_{beta}_{repeat}"
            filled_script = TEMPLATE.format(
                **{"name": name, "beta": beta, "iter": repeat}
            )
            with open(name + ".slurm", "w") as fh:
                fh.write(filled_script)

            if submit:
                subprocess.call(
                    "sbatch {}".format(name + ".slurm"), shell=True, cwd="."
                )
                time.sleep(5)


if __name__ == "__main__":
    main()
