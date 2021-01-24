# -*- coding: utf-8 -*-
"""
Functions that are used while a Generation is being Evaluated
"""
import os
from random import randrange

import numpy as np
from guacamol.scoring_function import ScoringFunction
from numpy.core.arrayprint import get_printoptions
from rdkit import Chem
from rdkit.Chem import Draw

from ...net import discriminator as D
from ...net import evolution_functions as evo
from ...sa_scorer.sascorer import calculate_score


def fitness(
    molecules_here,
    properties_calc_ls,
    discriminator,
    disc_enc_type,
    generation_index,
    max_molecules_len,
    device,
    writer,
    beta,
    data_dir,
    scoring_function,
    max_fitness_collector,
    watchtime: int = 5,
    similarity_threshold: float = 0.2,
    num_processors=1,
):
    """Calculate fitness fo a generation in the GA

    All properties are standardized based on the mean & stddev of the zinc dataset

    Parameters:
    molecules_here    (list)         : List of a string of molecules
    properties_calc_ls               : List of properties to calculate
    discriminator     (torch.Model)  : Pytorch classifier
    disc_enc_type     (string)       : Indicated type of encoding shown to discriminator
    generation_index  (int)          : Which generation indicator
    max_molecules_len (int)          : Largest mol length
    device            (string)       : Device of discrimnator
    max_fitness_collector (list)     : Maximum fitnesses so far for adaptive penalty

    Returns:
    fitness                   (np.array) : A lin comb of properties and
                                           discriminator predictions
    discriminator_predictions (np.array) : The predictions made by the discrimantor

    """
    beta_ = 0
    dataset_x = evo.obtain_discr_encoding(
        molecules_here,
        disc_enc_type,
        max_molecules_len,
        num_processors,
        generation_index,
    )
    if generation_index == 1:
        discriminator_predictions = np.zeros((len(dataset_x), 1))
    else:
        discriminator_predictions = D.do_predictions(discriminator, dataset_x, device)

    discriminator_predictions = discriminator_predictions.flatten()
    molecules_here_unique = list(set(molecules_here))

    scores = np.array(scoring_function.score_list(molecules_here))

    max_fitness_collector.append(np.max(scorƒes))

    # Impose the beta cutoff!
    if generation_index > 20:
        if len(set(max_fitness_collector[-watchtime:])) == 1:
            beta_ = beta
            print(f"BETA CUTTOFF IMPOSED {beta_} index: ", generation_index)

    fitness = (beta_ * discriminator_predictions) + scores

    # Plot fitness with discriminator
    writer.add_scalar("max fitness with discrm", max(fitness), generation_index)
    writer.add_scalar("avg fitness with discrm", fitness.mean(), generation_index)

    return (fitness, scores, discriminator_predictions, molecules_here_unique)


def obtain_fitness(
    smiles_here,
    selfies_here,
    properties_calc_ls,
    discriminator,
    disc_enc_type,
    generation_index,
    max_molecules_len,
    device,
    writer,
    beta,
    data_dir,
    image_dir,
    scoring_function,
    max_fitness_collector,
    watchtime: int = 5,
    similarity_threshold: float = 0.2,
    num_processors=1,
):
    """Obtain fitness of generation based on choices of disc_enc_type.
    Essentially just calls 'fitness'
    """

    (fitness_here, scores, discriminator_predictions, _) = fitness(
        molecules_here=smiles_here,
        properties_calc_ls=properties_calc_ls,
        discriminator=discriminator,
        disc_enc_type=disc_enc_type,
        generation_index=generation_index,
        max_molecules_len=max_molecules_len,
        device=device,
        writer=writer,
        beta=beta,
        data_dir=data_dir,
        scoring_function=scoring_function,
        max_fitness_collector=max_fitness_collector,
        watchtime=watchtime,
        similarity_threshold=similarity_threshold,
        num_processors=num_processors,
    )

    # Now, order the things ...
    (
        order,
        fitness_ordered,
        smiles_ordered,
        selfies_ordered,
        scores_ordered,
        discriminator_ordered,
    ) = order_based_on_fitness(
        fitness_here, smiles_here, selfies_here, scores, discriminator_predictions
    )

    # print statement for the best molecule in the generation
    print("Best best molecule in generation ", generation_index)
    print("    smile  : ", smiles_ordered[0])
    print("    fitness: ", fitness_ordered[0])
    print("    discrm : ", discriminator_ordered[0])

    show_generation_image(
        generation_index,
        image_dir,
        smiles_ordered,
        fitness_ordered,
        discriminator_predictions,
    )

    return (
        order,
        fitness_ordered,
        smiles_ordered,
        selfies_ordered,
        scores_ordered,
        discriminator_predictions[0],
    )


def show_generation_image(
    generation_index, image_dir, smiles_ordered, fitness, discr_scores,
):
    """Plot 100 molecules with the best fitness in in a generation
        Called after at the end of each generation. Image in each generation
        is stored with name 'generation_index.png'

    Images are stored in diretory './images'
    """
    if generation_index > 1:
        A = list(smiles_ordered)
        A = A[:100]
        if len(A) < 100:
            return  # raise Exception('Not enough molecules provided for plotting ', len(A))
        A = [Chem.MolFromSmiles(x) for x in A]

        create_100_mol_image(
            A,
            os.path.join(image_dir, str(generation_index) + "_ga.png"),
            fitness,
            discr_scores,
        )


def order_based_on_fitness(
    fitness_here, smiles_here, selfies_here, scores_here, discriminator_predictions
):
    """Order elements of a lists (args) based om Decreasing fitness"""
    order = np.argsort(scores_here)[
        ::-1
    ]  # Decreasing order of indices, based on fitness
    fitness_ordered = [fitness_here[idx] for idx in order]
    smiles_ordered = [smiles_here[idx] for idx in order]
    selfies_ordered = [selfies_here[idx] for idx in order]
    scores_ordered = [scores_here[idx] for idx in order]
    discriminator_ordered = [discriminator_predictions[idx] for idx in order]
    return (
        order,
        fitness_ordered,
        smiles_ordered,
        selfies_ordered,
        scores_ordered,
        discriminator_ordered,
    )


def create_100_mol_image(mol_list, file_name, fitness, discr_scores):
    """Create a single picture of multiple molecules in a single Grid - with properties underneath."""
    assert len(mol_list) == 100

    for i, m in enumerate(mol_list):
        m.SetProp(
            "_Name", "%s %s " % (round(fitness[i], 3), round(discr_scores[i], 3),),
        )
    try:
        Draw.MolsToGridImage(
            mol_list,
            molsPerRow=10,
            subImgSize=(200, 200),
            legends=[x.GetProp("_Name") for x in mol_list],
        ).save(file_name)
    except Exception as e:
        print("Failed to produce image due to {}".format(e))
    return


def obtain_previous_gen_mol(
    starting_smiles,
    starting_selfies,
    generation_size,
    generation_index,
    selfies_all,
    smiles_all,
):
    """Obtain molecules from one generation prior.
    If generation_index is 1, only the the starting molecules are returned
    """
    if generation_index == 1:
        randomized_smiles = []
        randomized_selfies = []
        for i in range(generation_size):  # nothing to obtain from previous gen
            # So, choose random moleclues from the starting list
            index = randrange(len(starting_smiles))
            randomized_smiles.append(starting_smiles[index])
            randomized_selfies.append(starting_selfies[index])

        return randomized_smiles, randomized_selfies
    else:
        return smiles_all[generation_index - 2], selfies_all[generation_index - 2]


def apply_generation_cutoff(order, generation_size):
    """Return of a list of indices of molecules that are kept (high fitness)
        and a list of indices of molecules that are replaced   (low fitness)

    The cut-off is imposed using a Fermi-Function

    Parameters:
    order (list)          : list of molecule indices arranged in Decreasing order of fitness
    generation_size (int) : number of molecules in a generation

    Returns:
    to_replace (list): indices of molecules that will be replaced by random mutations of
                       molecules in list 'to_keep'
    to_keep    (list): indices of molecules that will be kept for the following generations
    """
    # Get the probabilities that a molecule with a given fitness will be replaced
    # a fermi function is used to smoothen the transition
    positions = np.array(range(0, len(order))) - 0.2 * float(len(order))
    probabilities = 1.0 / (
        1.0 + np.exp(-0.02 * generation_size * positions / float(len(order)))
    )

    to_replace = []  # all molecules that are replaced
    to_keep = []  # all molecules that are kept
    for idx in range(0, len(order)):
        if np.random.rand(1) < probabilities[idx]:
            to_replace.append(idx)
        else:
            to_keep.append(idx)

    return to_replace, to_keep


def obtain_next_gen_molecules(
    order, to_replace, to_keep, selfies_ordered, smiles_ordered, max_molecules_len
):
    """Obtain the next generation of molecules. Bad molecules are replaced by
    mutations of good molecules

    Parameters:
    order (list)            : list of molecule indices arranged in Decreasing order of fitness
    to_replace (list)       : list of indices of molecules to be replaced by random mutations of better molecules
    to_keep (list)          : list of indices of molecules to be kept in following generation
    selfies_ordered (list)  : list of SELFIE molecules, ordered by fitness
    smiles_ordered (list)   : list of SMILE molecules, ordered by fitness
    max_molecules_len (int) : length of largest molecule


    Returns:
    smiles_mutated (list): next generation of mutated molecules as SMILES
    selfies_mutated(list): next generation of mutated molecules as SELFIES
    """
    smiles_mutated = []
    selfies_mutated = []
    for idx in range(0, len(order)):
        if idx in to_replace:  # smiles to replace (by better molecules)
            random_index = np.random.choice(to_keep, size=1, replace=True, p=None)[
                0
            ]  # select a random molecule that survived
            grin_new, smiles_new = evo.mutations_random_grin(
                selfies_ordered[random_index], max_molecules_len
            )  # do the mutation

            # add mutated molecule to the population
            smiles_mutated.append(smiles_new)
            selfies_mutated.append(grin_new)
        else:  # smiles to keep
            smiles_mutated.append(smiles_ordered[idx])
            selfies_mutated.append(selfies_ordered[idx])
    return smiles_mutated, selfies_mutated


def obtain_discrm_data(
    disc_enc_type,
    molecules_reference,
    smiles_mutated,
    selfies_mutated,
    max_molecules_len,
    num_processors,
    generation_index,
):
    """Obtain data that will be used to train the discriminator (inputs & labels)"""
    if disc_enc_type == "smiles":
        random_dataset_selection = np.random.choice(
            list(molecules_reference.keys()), size=len(smiles_mutated)
        ).tolist()
        dataset_smiles = (
            smiles_mutated + random_dataset_selection
        )  # Generation smiles + Dataset smiles
        dataset_x = evo._to_onehot(dataset_smiles, disc_enc_type, max_molecules_len)
        dataset_y = np.array(
            [1 if x in molecules_reference else 0 for x in smiles_mutated]
            + [1 for i in range(len(dataset_smiles) - len(smiles_mutated))]
        )

    elif disc_enc_type == "selfies":
        random_dataset_selection = np.random.choice(
            list(molecules_reference.keys()), size=len(selfies_mutated)
        ).tolist()
        dataset_smiles = selfies_mutated + random_dataset_selection
        dataset_x = evo._to_onehot(dataset_smiles, disc_enc_type, max_molecules_len)
        dataset_y = np.array(
            [1 if x in molecules_reference else 0 for x in selfies_mutated]
            + [1 for i in range(len(dataset_smiles) - len(selfies_mutated))]
        )

    elif disc_enc_type == "properties_rdkit":
        random_dataset_selection = np.random.choice(
            list(molecules_reference.keys()), size=len(smiles_mutated)
        ).tolist()
        dataset_smiles = (
            smiles_mutated + random_dataset_selection
        )  # Generation smiles + Dataset smiles
        dataset_x = evo.obtain_discr_encoding(
            dataset_smiles,
            disc_enc_type,
            max_molecules_len,
            num_processors,
            generation_index,
        )
        dataset_y = np.array(
            [1 if x in molecules_reference else 0 for x in smiles_mutated]
            + [1 for i in range(len(dataset_smiles) - len(selfies_mutated))]
        )

    # Shuffle training data
    order_training = np.array(
        range(len(dataset_smiles))
    )  # np.arange(len(dataset_smiles))
    np.random.shuffle(order_training)
    dataset_x = dataset_x[order_training]
    dataset_y = dataset_y[order_training]

    return dataset_x, dataset_y


def update_gen_res(
    smiles_all, smiles_mutated, selfies_all, selfies_mutated, smiles_all_counter
):
    """Collect results that will be shared with global variables outside generations"""
    smiles_all.append(smiles_mutated)
    selfies_all.append(selfies_mutated)

    for smi in smiles_mutated:
        if smi in smiles_all_counter:
            smiles_all_counter[smi] += 1
        else:
            smiles_all_counter[smi] = 1

    return smiles_all, selfies_all, smiles_all_counter
