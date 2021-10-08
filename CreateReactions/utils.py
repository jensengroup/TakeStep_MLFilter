import copy
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem.EnumerateStereoisomers import StereoEnumerationOptions

import take_step
import xyz2mol


def Nhack(smiles):
    """Remove chiral tag on N"""
    if "N@@" in smiles:
        smiles = smiles.replace("N@@", "N")
    if "N@" in smiles:
        smiles = smiles.replace("N@", "N")
    return smiles

def mapped_smi_to_mol(smi):
    molobj = Chem.MolFromSmiles(smi, sanitize=False)
    Chem.SanitizeMol(molobj)
    return take_step.reassign_atom_idx(molobj)


def ac2smiles(ac, atoms, charge, protomol):
    product_mol = xyz2mol.AC2mol(protomol, ac, atoms, charge, use_atom_maps=True)
    if Chem.rdmolops.GetFormalCharge(product_mol) != charge:
        return np.nan, np.nan

    mapped_product_smi = Chem.MolToSmiles(product_mol)

    # Remove atom mapping
    [atom.SetAtomMapNum(0) for atom in product_mol.GetAtoms()]
    product_smi = Chem.MolToSmiles(product_mol, allHsExplicit=True)

    return mapped_product_smi, product_smi


def run_step(rsmi_init, max_bond=3, cd=5):
    """ """
    opts = StereoEnumerationOptions(onlyUnassigned=False, unique=False)

    rmol = mapped_smi_to_mol(rsmi_init)
    rdmolops.AssignStereochemistry(rmol, flagPossibleStereoCenters=True, force=True)
    rmol = next(EnumerateStereoisomers(rmol, options=opts))
    pmols = list(take_step.TakeElementaryStep(rmol, max_bond, cd).get_products())

    mapped_rsmi = Chem.MolToSmiles(rmol)
    copy_rmol = copy.deepcopy(rmol)
    [atom.SetAtomMapNum(0) for atom in copy_rmol.GetAtoms()]
    rdmolops.AssignStereochemistry(copy_rmol, force=True, cleanIt=True)
    rsmi = Nhack(Chem.MolToSmiles(copy_rmol))

    mapped_psmis = []
    psmis = []
    for pmol in copy.deepcopy(pmols):
        mapped_psmi = Chem.MolToSmiles(pmol)
        [atom.SetAtomMapNum(0) for atom in pmol.GetAtoms()]
        psmi = Nhack(Chem.MolToSmiles(pmol))

        mapped_psmis.append(mapped_psmi)
        psmis.append(psmi)

    # Make DF
    product_dataframe = pd.DataFrame(
        zip([rsmi_init] * len(pmols), [rmol] * len(pmols), pmols),
        columns=["rsmi_org", "rmol", "pmol"],
    )
    product_dataframe["mapped_rsmi"] = [mapped_rsmi] * len(pmols)
    product_dataframe["mapped_psmi"] = mapped_psmis
    product_dataframe["rsmi"] = [rsmi] * len(pmols)
    product_dataframe["psmi"] = psmis

    return product_dataframe
