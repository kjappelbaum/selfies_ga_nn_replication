import os

import wandb


def read_dataset(filename):
    """Return a list of smiles contained in file filename

    Parameters:
    filename (string) : Name of file containg smiles seperated by '\n'

    Returns
    content  (list)   : list of smile string in file filename
    """
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    return content


def read_dataset_encoding(disc_enc_type, run):
    """Return zinc-data set based on disc_enc_type choice of 'smiles' or 'selfies'

    Parameters:
    disc_enc_type (string): 'smiles' or 'selfies'
    """
    if disc_enc_type == "smiles" or disc_enc_type == "properties_rdkit":

        artifact = run.use_artifact("zinc_dearom:latest")
        artifact_dir = artifact.download()
        smiles_reference = read_dataset(
            filename=os.path.join(artifact_dir, "zinc_dearom.txt")
        )
        return smiles_reference
    elif disc_enc_type == "selfies":
        artifact = run.use_artifact("SELFIES_zinc:latest")
        artifact_dir = artifact.download()
        selfies_reference = read_dataset(
            filename=os.path.join(artifact_dir, "SELFIES_zinc.txt")
        )
        return selfies_reference


def create_artifacts():
    run = wandb.init(project="ga_replication_study", job_type="upload")
    zinc_dearom_raw = wandb.Artifact("zinc_dearom", type="raw_data")
    zinc_dearom_raw.add_file("zinc_dearom.txt")
    selfies_zinc_raw = wandb.Artifact("SELFIES_zinc", type="raw_data")
    selfies_zinc_raw.add_file("SELFIES_zinc.txt")
    run.log_artifact(zinc_dearom_raw)
    run.log_artifact(selfies_zinc_raw)


if __name__ == "__main__":
    create_artifacts()