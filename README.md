# Replication study of "Augmenting genetic algorithms with deep neural networks for exploring the chemical space"

|                    |                                                                                                                                                                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Original Reference | [AkshatKumar Nigam, Pascal Friederich, Mario Krenn, Alan Aspuru-Guzik, ICLR 2020](https://openreview.net/forum?id=H1lmyRNFvr)                                                                                                        |
| Original Code      | [On GitHub](https://github.com/aspuru-guzik-group/GA/tree/paper_results)                                                                                                                                                          |
| All experiments    | [Tracked using wandb](https://wandb.ai/kjappelbaum/ga_replication_study)                                                                                                                                                             |
| Interactive Report | [On the Weights and Biases platform](https://wandb.ai/kjappelbaum/ga_replication_study/reports/A-reproducibility-study-of-Augmenting-Genetic-Algorithms-with-Deep-Neural-Networks-for-Exploring-the-Chemical-Space--Vmlldzo0MjI5NjI) |

## Requirements

The main dependencies for the GA are [PyTorch](https://pytorch.org/), [RDKit](rdkit.org), and [SELFIES](https://github.com/aspuru-guzik-group/selfies). For the similarity-triggered penalty we additionally used [ccbmlib](https://github.com/vogt-m/ccbmlib).

### Linux

To create a conda environment with all dependencies run

```bash
conda env create -n selfies_ga_replication -f environment_ubuntu.yml
```

### Mac OS

To create a conda environment with all dependencies run

```bash
conda env create -n selfies_ga_replication -f environment_mac.yml
```

(We used the Mac environment mostly for data analysis.)

## Experiments

In addition to the main experiments from the paper we ran some additional ones investigating a novel adapative penalty, the influence of the model architecture, and the influence of the labeling convention.

1. [Experiment 1: Baseline and basic SELFIES GA](experiments/experiment_1)
2. [Experiment 2: Time adaptive penalty](experiments/experiment_2)
3. [Experiment 3: Unsupervised analysis (clustering)](experiments/experiment_3) (not shown in the report)
4. [Experiment 4: Constrained optimization](experiments/experiment_4)
5. [Experiment 5: Simultaneous logP and QED optimization](experiments/experiment_5)
6. [Experiment 6: Flipping the objective](experiments/experiment_6)
7. [Experiment 7: Similarity-triggered adpative penalty](experiments/experiment_7)
8. [Experiment 8: Using logistic regression](experiments/experiment_8)
9. [Guacamol benchmark](experiments/guacamol_baseline)

We reran the experiments using [Weights and Biases](https://wandb.ai/site) tracking, if you also want to use this, you'll need to setup wandb on your machine (`pip install wandb`, followed by `wandb login`). Typically, we ran experiments in the following way from the root directory

```bash
python -m experiments.experiment_1.ga.core_ga
```

We ran some additional experiments on a cluster. For this we used the submission scripts in `submission_scripts`.

### Analysis

The analysis was performed in Jupyter notebook which are in a `analysis` subfolder for every experiment.
