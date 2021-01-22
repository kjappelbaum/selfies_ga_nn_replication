import os
import multiprocessing
import time
import click
import torch
from typing import List
from joblib import delayed
from selfies import decoder, encoder
from tensorboardX import SummaryWriter

import wandb

from ...datasets.prepare_data import read_dataset_encoding
from ...net import discriminator as D
from ...net import evolution_functions as evo
from . import generation_props_no_time as gen_func

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from guacamol.utils.chemistry import canonicalize


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
    beta,
    run,
    table,
    scoring_function,
    image_dir,
    writer,
    data_dir,
    saved_models_dir,
    max_fitness_collector,
    watchtime=5,
    similarity_threshold=0.2,
    num_processors=1,
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

        # Calculate fitness of previous generation
        (
            order,
            fitness_ordered,
            smiles_ordered,
            selfies_ordered,
            scores_ordered,
            discriminator_predictions,
        ) = gen_func.obtain_fitness(
            smiles_here=smiles_here,
            selfies_here=selfies_here,
            properties_calc_ls=properties_calc_ls,
            discriminator=discriminator,
            disc_enc_type=disc_enc_type,
            generation_index=generation_index,
            max_molecules_len=max_molecules_len,
            device=device,
            writer=writer,
            beta=beta,
            data_dir=data_dir,
            image_dir=image_dir,
            scoring_function=scoring_function,
            watchtime=watchtime,
            similarity_threshold=similarity_threshold,
            num_processors=num_processors,
            max_fitness_collector=max_fitness_collector,
        )

        run.log(
            {
                "fitness": fitness_ordered[0],
                "score": scores_ordered[0],
                "discriminator": discriminator_predictions,
            }
        )
        table.add_data(
            generation_index,
            smiles_ordered[0],
            selfies_ordered[0],
            fitness_ordered[0],
            scores_ordered[0],
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
    return smiles_ordered


class ChemGEGenerator(GoalDirectedGenerator):
    def __init__(
        self,
        run,
        beta=0,
        watchtime=5,
        similarity_threshold=0.4,
        num_generations=500,
        generation_size=1000,
        max_molecule_len=81,
        disc_epoch_per_gen=10,
        disc_enc_type="properties_rdkit",
        disc_layers=[100, 51],
        training_start_gen=0,
        device="cpu",
        properties_calc_ls=None,
        results_path="results",
    ):
        self.num_generation = num_generations
        self.generation_size = generation_size
        self.max_molecule_len = max_molecule_len
        self.disc_epoch_per_gen = disc_epoch_per_gen
        self.disc_enc_type = disc_enc_type
        self.disc_layers = disc_layers
        self.training_start_gen = training_start_gen
        self.device = device
        self.properties_calc_ls = properties_calc_ls
        self.beta = beta
        self.watchtime = watchtime
        self.similarity_threshold = similarity_threshold
        self.num_processors = 1
        self.run = run
        self.table = wandb.Table(
            columns=[
                "generation",
                "SMILES",
                "SELFIES",
                "fitness",
                "score",
                "discriminator",
            ]
        )
        self.results_dir = evo.make_clean_results_dir(
            os.path.join(THIS_DIR, results_path)
        )
        (
            self.image_dir,
            self.saved_models_dir,
            self.data_dir,
        ) = evo.make_clean_directories(beta, self.results_dir, "")
        self.max_fitness_collector = []

        torch.cuda.empty_cache()
        self.writer = SummaryWriter()

    def load_smiles_from_file(
        self, smi_file=os.path.join(THIS_DIR, "..", "data", "guacamol_v1_all.smiles")
    ):
        with open(smi_file) as f:
            return [canonicalize(s.strip()) for s in f.readlines()]

    def top_k(self, smiles: List[str], scoring_function: ScoringFunction, k):
        scores = scoring_function.score_list(smiles)
        scored_smiles = list(zip(scores, smiles))
        scored_smiles = sorted(scored_smiles, key=lambda x: x[0], reverse=True)
        return [smile for _, smile in scored_smiles][:k]

    def generate_optimized_molecules(
        self,
        starting_population: List[str],
        scoring_function: ScoringFunction,
        number_molecules: int,
    ) -> List[str]:
        """Returns a list of SMILES"""
        if starting_population is not None:
            starting_selfies = [encoder(s) for s in starting_population]
        else:
            starting_population = self.top_k(
                self.load_smiles_from_file(), scoring_function, self.generation_size
            )

            starting_selfies = [encoder(s) for s in starting_population]

        best_smiles = initiate_ga(
            num_generations=self.num_generation,
            generation_size=self.generation_size,
            starting_selfies=starting_selfies,
            max_molecules_len=self.max_molecule_len,
            disc_epochs_per_generation=self.disc_epoch_per_gen,
            disc_enc_type=self.disc_enc_type,
            disc_layers=self.disc_layers,
            training_start_gen=self.training_start_gen,
            device=self.device,
            properties_calc_ls=self.properties_calc_ls,
            beta=self.beta,
            run=self.run,
            table=self.table,
            scoring_function=scoring_function,
            image_dir=self.image_dir,
            writer=self.writer,
            data_dir=self.data_dir,
            saved_models_dir=self.saved_models_dir,
            max_fitness_collector=self.max_fitness_collector,
            watchtime=self.watchtime,
            similarity_threshold=self.similarity_threshold,
            num_processors=self.num_processors,
        )

        return best_smiles[:number_molecules]


@click.command("cli")
@click.argument("beta", type=float, default=0)
@click.argument("watchtime", type=int, default=5)
@click.argument("similarity_threshold", type=float, default=0.5)
@click.argument("output_dir", type=click.Path(), default="results")
def cli(beta, watchtime, similarity_threshold, output_dir):
    num_generations = 100
    generation_size = 500
    max_molecule_len = 81
    disc_epoch_per_gen = 10
    disc_enc_type = "properties_rdkit"
    disc_layers = [100, 51]
    training_start_gen = 0
    properties_calc_ls = [
        "logP",
        "SAS",
        "RingP",
    ]

    run = wandb.init(
        project="ga_replication_study",
        tags=["ga", "guacamol_benchmark", "no_adaptive"],
        config={
            "beta": beta,
            "alphabet": "original",
            "num_generations": num_generations,
            "generation_size": generation_size,
            "max_molecules_len": max_molecule_len,
            "disc_epochs_per_generation": disc_epoch_per_gen,
            "disc_enc_type": disc_enc_type,
            "disc_layers": disc_layers,
            "training_start_gen": training_start_gen,
            "properties_calc_ls": properties_calc_ls,
        },
        reinit=True,
    )

    print(run.config)

    optimiser = ChemGEGenerator(
        run=run,
        beta=beta,
        watchtime=watchtime,
        similarity_threshold=similarity_threshold,
        # results_path=output_dir,
        num_generations=num_generations,
        generation_size=generation_size,
        max_molecule_len=max_molecule_len,
        disc_epoch_per_gen=disc_epoch_per_gen,
        disc_enc_type=disc_enc_type,
        disc_layers=disc_layers,
        training_start_gen=training_start_gen,
        properties_calc_ls=properties_calc_ls,
    )

    json_file_path = os.path.join(THIS_DIR, f"goal_directed_results_beta_{beta}.json")
    assess_goal_directed_generation(optimiser, json_output_file=json_file_path)


if __name__ == "__main__":
    cli()
