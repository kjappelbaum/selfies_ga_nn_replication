import os
import multiprocessing
import time
import click
import torch
from selfies import decoder, encoder
from tensorboardX import SummaryWriter

import wandb

from ...datasets.prepare_data import read_dataset_encoding
from ...net import discriminator as D
from ...net import evolution_functions as evo
from . import generation_props as gen_func

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def initiate_ga(
    num_generations,
    generation_size,
    starting_selfies,
    max_molecules_len,
    disc_epochs_per_generation,
    disc_enc_type,
    disc_layers,
    training_start_gen,
    device,
    properties_calc_ls,
    num_processors,
    beta,
    run,
    table,
    scoring_function,
    max_fitness_collector,
    watchtime=5,
    similarity_threshold=0.2,
):

    # Obtain starting molecule
    starting_smiles = evo.sanitize_multiple_smiles(
        [decoder(selfie) for selfie in starting_selfies]
    )

    # Recording Collective results
    smiles_all = []  # all SMILES seen in all generations
    selfies_all = []  # all SELFIES seen in all generation
    smiles_all_counter = {}  #

    # Initialize a Discriminator
    discriminator, d_optimizer, d_loss_func = D.obtain_initial_discriminator(
        disc_enc_type, disc_layers, max_molecules_len, device
    )

    wandb.watch(discriminator)

    # Read in the Zinc data set
    molecules_reference = read_dataset_encoding(disc_enc_type, run)
    # convert the zinc data set into a dictionary
    molecules_reference = dict.fromkeys(molecules_reference, "")

    # Set up Generation Loop
    total_time = time.time()
    for generation_index in range(1, num_generations + 1):
        print("###   On generation %i of %i" % (generation_index, num_generations))
        start_time = time.time()

        # Obtain molecules from the previous generation
        smiles_here, selfies_here = gen_func.obtain_previous_gen_mol(
            starting_smiles,
            starting_selfies,
            generation_size,
            generation_index,
            selfies_all,
            smiles_all,
        )

        # Calculate fitness of previous generation (shape: (generation_size, ))
        (
            fitness_here,
            order,
            fitness_ordered,
            smiles_ordered,
            selfies_ordered,
            fitness_no_discriminator,
            discriminator_predictions,
        ) = gen_func.obtain_fitness(
            disc_enc_type,
            smiles_here,
            selfies_here,
            properties_calc_ls,
            discriminator,
            generation_index,
            max_molecules_len,
            device,
            generation_size,
            num_processors,
            writer,
            beta,
            image_dir,
            data_dir,
            max_fitness_collector,
            watchtime,
            similarity_threshold,
        )

        run.log(
            {
                "fitness": fitness_ordered[0],
                "fitness_no_discr": fitness_no_discriminator,
                "discriminator": discriminator_predictions,
            }
        )
        table.add_data(
            generation_index,
            smiles_ordered[0],
            selfies_ordered[0],
            fitness_ordered[0],
            fitness_no_discriminator,
            discriminator_predictions,
        )
        # Obtain molecules that need to be replaced & kept
        to_replace, to_keep = gen_func.apply_generation_cutoff(order, generation_size)

        # Obtain new generation of molecules
        smiles_mutated, selfies_mutated = gen_func.obtain_next_gen_molecules(
            order,
            to_replace,
            to_keep,
            selfies_ordered,
            smiles_ordered,
            max_molecules_len,
        )
        # Record in collective list of molecules
        smiles_all, selfies_all, smiles_all_counter = gen_func.update_gen_res(
            smiles_all, smiles_mutated, selfies_all, selfies_mutated, smiles_all_counter
        )

        # Obtain data for training the discriminator (Note: data is shuffled)
        dataset_x, dataset_y = gen_func.obtain_discrm_data(
            disc_enc_type,
            molecules_reference,
            smiles_mutated,
            selfies_mutated,
            max_molecules_len,
            num_processors,
            generation_index,
        )
        # Train the discriminator (on mutated molecules)
        if generation_index >= training_start_gen:
            discriminator = D.do_x_training_steps(
                dataset_x,
                dataset_y,
                discriminator,
                d_optimizer,
                d_loss_func,
                disc_epochs_per_generation,
                generation_index - 1,
                device,
                writer,
                data_dir,
            )
            modelpath = D.save_model(
                discriminator, generation_index - 1, saved_models_dir
            )  # Save the discriminator

            artifact = wandb.Artifact(f"model_{generation_index}", type="model")
            artifact.add_file(modelpath)
            run.log_artifact(artifact)

        print("Generation time: ", round((time.time() - start_time), 2), " seconds")

    print("Total time: ", round((time.time() - total_time) / 60, 2), " mins")
    print("Total number of unique molecules: ", len(smiles_all_counter))
    return smiles_all


class ChemGEGenerator(GoalDirectedGenerator):

    def __init__(self, beta=0, watchtime=5, starting_smiles="C", similarity_threshold=0.4,num_generations=500,  generation_size=1000, max_molecule_len=81, disc_epoch_per_gen=10, disc_enc_type='properties_rdkit', disc_layers=[100,51], training_start_gen=0, device='cpu', properties_calc_ls=None):
        self.starting_selfies = [encoder(starting_smile)]
        self.num_generation = num_generations
        self.generation_size = generation_size
        self.max_molecule_len = max_molecule_len
        self.disc_epoch_per_gen = disc_epoch_per_gen
        self.disc_enc_type = disc_enc_type
        self.disc_layers = disc_layers
        self.training_start_gen = training_start_gen 
        self.device = device 
        self.properties_calc_ls = properties_calc_ls
        self.beta=beta
        self.watchtime=watchtime
        self.similarity_threshold =similarity_threshold 

    def top_k(self, smiles, scoring_function, k):
        joblist = (delayed(scoring_function.score)(s) for s in smiles)
        scores = self.pool(joblist)
        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for score, smile in scored_smiles][:k]


    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                    starting_population: Optional[List[str]] = None) -> List[str]:

        """Returns a list of SMILES"""
        best_smiles = initiate_ga()