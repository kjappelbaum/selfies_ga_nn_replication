# -*- coding: utf-8 -*-
import ast

import ccbmlib.models as ccbm

import wandb

api = wandb.Api()
import sys

sys.path.append("..")
from collections import OrderedDict

import numpy as np
from net import evolution_functions as evo
from sa_scorer.sascorer import calculate_score


def get_pairwise_similarities(mols):
    pair_wise_similarities_random_baseline = []

    for i, mol0 in enumerate(mols):
        for j, mol1 in enumerate(mols):
            if i < j:
                m0 = ccbm.morgan(mol0, radius=2)
                m1 = ccbm.morgan(mol1, radius=2)
                pair_wise_similarities_random_baseline.append(ccbm.tc(m0, m1))

    return pair_wise_similarities_random_baseline


def get_smiles_sizes(run_id):
    sizes = []
    mols = []
    run = api.run(f"kjappelbaum/ga_replication_study/{run_id}")

    def get_file(run):
        for file in run.files():
            if "media/table/Table" in str(file):
                return file
        return None

    file = get_file(run)

    if file:
        path = file.download(replace=True)
        lines = path.readlines()
        smiles = []

        for i in ast.literal_eval(lines[0])["data"]:
            smiles.append(i[1])

        for smile in smiles:
            mol, _, _ = evo.sanitize_smiles(smile)
            sizes.append(mol.GetNumHeavyAtoms())
            mols.append(mol)

    return sizes, mols


def get_smiles_sizes_2(run_id):
    sizes = []
    mols = []
    js = []
    run = api.run(f"kjappelbaum/ga_replication_study/{run_id}")

    def get_file(run):
        for file in run.files():
            if "media/table/Table" in str(file):
                return file
        return None

    file = get_file(run)

    if file:
        path = file.download(replace=True)
        lines = path.readlines()
        smiles = []

        for i in ast.literal_eval(lines[0])["data"]:
            smiles.append(i[1])
            js.append(i[-1])

        for smile in smiles:
            mol, _, _ = evo.sanitize_smiles(smile)
            sizes.append(mol.GetNumHeavyAtoms())
            mols.append(mol)

    return sizes, mols, js


def get_similarity_evolution(mols):
    similarities = []

    for i in range(5, 500):
        similarities.append(np.mean(get_pairwise_similarities(mols[i - 5 : i])))
    return similarities


def get_similarity_size_evolution(run_name):
    sizes, mols = get_smiles_sizes(run_name)

    result = OrderedDict(
        [
            ("mols", mols),
            ("sizes", sizes),
            ("similarity", get_similarity_evolution(mols)),
        ]
    )

    return result
