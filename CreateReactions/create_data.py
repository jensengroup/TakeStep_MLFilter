import os
import glob
import random
import pandas as pd
from tqdm import tqdm as tqdm
from joblib import Parallel, delayed

import utils

import rdkit

random.seed(31)


print("RDKit version: ", rdkit.__version__)


def reac_eq_prod(rsmi, psmi):
    """Remove reactions where nothing but chitality have changed"""
    return rsmi.replace("@", "") == psmi.replace("@", "")


def store_data(df, fname):
    """
    Write products to csv file
    """
    df.to_pickle(fname + ".pkl")
    df[["rsmi_org", "mapped_rsmi", "mapped_psmi", "rsmi", "psmi"]].to_csv(
        fname + ".csv", index=False
    )


def run_reaction(smi, ridx):
    """
    Run TakeStep code. Break and form 3 bonds but max 5 changes.
    """
    try:
        reaction_df = utils.run_step(smi)
        store_data(reaction_df, f"data/output/reaction_idx_{ridx}")
    except KeyError:
        print(smi)


def merge_product_data():
    """Merge all reactions into one DF"""
    output_files = glob.glob("data/output/*csv")
    dfs = []
    for _file in output_files:
        idx = _file.split("/")[-1].split(".")[0].split("_")[-1]
        df = pd.read_csv(_file)

        fragments = df["psmi"].str.split(".", expand=True)
        df = pd.concat([df, fragments], axis=1)
        df["reac_idx"] = [idx] * len(df)

        rename_dict = zip(fragments.columns, [f"frag{i}" for i in fragments.columns])
        df.rename(columns=dict(rename_dict), inplace=True)

        dfs.append(df)

    df_reactions = pd.concat(dfs)

    # Remove reactions where nothing but chirality changed.
    r_eq_p = []
    for rsmi, psmi in df_reactions[["rsmi", "psmi"]].itertuples(index=False):
        r_eq_p.append(reac_eq_prod(rsmi, psmi))

    df_reactions["reac_eq_prod"] = r_eq_p
    df_reactions = df_reactions[df_reactions["reac_eq_prod"] == False].copy()
    df_reactions.drop(columns=["reac_eq_prod"])

    df_reactions.to_csv("data/all_random_reactions.csv")

    return df_reactions


def unique_fragments(df):
    """Save all unique fragments for reactants and products"""
    # Find unique fragments
    frags_set = set()
    for i, x in enumerate(df[["frag0", "frag1", "frag2"]].to_numpy().flatten()):
        if not pd.isnull(x):
            frags_set.add(x)

    reactants_set = set(df_reactions.rsmi)

    print("# unique frags: ", len(frags_set))
    print("# unique reactants: ", len(reactants_set))

    pd.DataFrame(list(frags_set), columns=["fragments"]).to_csv(
        "mopac_fragments_input.csv", index=False
    )
    pd.DataFrame(list(reactants_set), columns=["fragments"]).to_csv(
        "mopac_reactants_input.csv", index=False
    )


if __name__ == "__main__":

    k_reactants = 4
    n_cpus = 4

    os.makedirs("data/output", exist_ok=True)

    # read data
    input_reactions = pd.read_csv("data/wb97xd3.csv", index_col="idx")
    random_reactants = random.choices(
        input_reactions["rsmi"].unique(), k=int(k_reactants)
    )

    # Make products
    idxs = []
    for rsmi in random_reactants:
        idxs.append(input_reactions[input_reactions["rsmi"] == rsmi].index[0])

    reac_input = list(zip(random_reactants, idxs))
    Parallel(n_jobs=n_cpus)(
        delayed(run_reaction)(smi, idx)
        for smi, idx in tqdm(reac_input, total=len(reac_input))
    )

    df_reactions = merge_product_data()
    unique_fragments(df_reactions)
