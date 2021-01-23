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
export WANDB_MODE=dryrun 
python -u -m experiments.experiment_7.ga.core_ga -- {beta} {tolerance} {watchtime} {iter}
"""

SIMILARITY = [0.2, 0.4, 0.8]
WATCHTIME = [5, 10, 50]
BETAS = [100, 500, 1000]
REPEATS = 5


@click.command("cli")
@click.option("--submit", is_flag=True)
def main(submit):
    for beta in BETAS:
        for watchtime in WATCHTIME:
            for similarity in SIMILARITY:
                for repeat in range(REPEATS):
                    repeat += 5
                    name = f"exp_7_{beta}_{watchtime}_{similarity}_{repeat}"
                    filled_script = TEMPLATE.format(
                        **{
                            "name": name,
                            "beta": beta,
                            "tolerance": similarity,
                            "watchtime": watchtime,
                            "iter": repeat,
                        }
                    )
                    with open(name + ".slurm", "w") as fh:
                        fh.write(filled_script)

                    if submit:
                        subprocess.call(
                            "sbatch {}".format(name + ".slurm"), shell=True, cwd="."
                        )
                        time.sleep(10)


if __name__ == "__main__":
    main()
