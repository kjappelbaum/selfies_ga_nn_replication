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
python -u -m experiments.guacamol_baseline.benchmarks.goal_benchmark_rev_ga-- {beta} {watchtime} {similarity_threshold} {num_generations}
"""

BETAS = [0, 10, 50, 100, 1000]
NUM_GENERATIONS = [10, 100, 200, 400]
SIMILARITY_THRESHOLD = [0.2, 0.4, 0.8]
WATCHTIME = [5, 10, 50]


def write_script_and_submit(
    beta, num_generations, watchtime, similarity_threshold, submit
):
    name = f"exp_guacamol_{beta}_{num_generations}_{watchtime}_{similarity_threshold}"
    filled_script = TEMPLATE.format(
        **{
            "name": name,
            "beta": beta,
            "watchtime": watchtime,
            "similarity_threshold": similarity_threshold,
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
                    for similarity_threshold in SIMILARITY_THRESHOLD:
                        write_script_and_submit(
                            beta,
                            num_generation,
                            watchtime,
                            watchtime,
                            similarity_threshold,
                            submit,
                        )
            else:
                watchtime = 0
                similarity_threshold = 0
                write_script_and_submit(
                    beta,
                    num_generation,
                    watchtime,
                    watchtime,
                    similarity_threshold,
                    submit,
                )


if __name__ == "__main__":
    main()
