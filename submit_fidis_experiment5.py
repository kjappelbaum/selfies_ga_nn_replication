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
python -u -m experiments.experiment_5.ga.core_ga {iter}
"""

REPEATS = 3


@click.command("cli")
@click.option("--submit", is_flag=True)
def main(submit):

    for repeat in range(REPEATS):
        repeat += 10
        name = f"exp_5_{repeat}"
        filled_script = TEMPLATE.format(**{"name": name, "iter": repeat})
        with open(name + ".slurm", "w") as fh:
            fh.write(filled_script)

        if submit:
            subprocess.call("sbatch {}".format(name + ".slurm"), shell=True, cwd=".")
            time.sleep(10)


if __name__ == "__main__":
    main()
