#!/groups/kemi/koerstz/anaconda3/envs/azqm/bin/python3

import numpy as np
import ppqm
from xyz2mol import xyz2AC_vdW
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdDistGeom, rdmolops


def write_xyz(atoms, coords, fname):
    """Write xyz string/file of qmconf"""

    xyz_string = str(len(atoms)) + "\n"
    xyz_string += "\n"

    for symbol, pos in zip(atoms, coords):
        xyz_string += "{}  {:10.5f} {:10.5f} {:10.5f}\n".format(symbol, *pos)

    with open(fname + ".xyz", "w") as xyz:
        xyz.write(xyz_string)


def make_conformers(molobj, max_confs=25):
    """ """
    rot_bond = rdMolDescriptors.CalcNumRotatableBonds(molobj)
    Chem.rdDistGeom.EmbedMultipleConfs(
        molobj,
        min(1 + 3 * rot_bond, max_confs),
        useRandomCoords=True,
        ETversion=2,
        maxAttempts=10_000,
    )


def check_connectivity(molobj, atoms, coords):
    """ """
    pt = Chem.GetPeriodicTable()
    old_adj = rdmolops.GetAdjacencyMatrix(molobj).astype(int)
    new_adj, _ = xyz2AC_vdW([pt.GetAtomicNumber(atom) for atom in atoms], coords)
    return np.array_equal(old_adj, new_adj)


def run_mopac_optimization(smiles, fname, max_confs=25):
    """ """
    molobj = Chem.MolFromSmiles(smiles, sanitize=False)
    Chem.SanitizeMol(molobj)
    make_conformers(molobj, max_confs)
    nconfs = molobj.GetNumConformers()

    if nconfs == 0:
        return nconfs, []

    mopac = ppqm.mopac.MopacCalculator(
        scr="_tmp_directory_",
        n_cores=1,
        cmd="/groups/kemi/koerstz/opt/mopac/mopac.sh",
        filename=fname + ".mop",
        options={},
    )
    mopac_options = {"PM3": None, "CYCLES": 256, "CHARGE": 0}
    if rdmolops.GetFormalCharge(molobj) % 2 != 0:  # if uneven electrons
        mopac_options["DOUBLET"] = None

    results = mopac.calculate(molobj, mopac_options)
    ok_confs = set()
    for i, res in enumerate(results):
        if check_connectivity(molobj, res["atoms"], res["coords"]):
            ok_confs.add(i)

    return nconfs, [res for i, res in enumerate(results) if i in ok_confs]


def can_converge(smi, fname):
    """ """
    molobj = Chem.MolFromSmiles(smi, sanitize=False)
    Chem.SanitizeMol(molobj)
    nconfs, res = run_mopac_optimization(smi, fname)
    energies = [r['h'] for r in res]
    return smi, nconfs, len(res), energies, len(res) > 0


if __name__ == "__main__":
    import pandas as pd
    import sys
    from joblib import Parallel, delayed

    inpname = sys.argv[1]
    outname = inpname.split('.')[0] + "_output.csv"

    input_smiles = pd.read_csv(inpname)['fragments'].tolist()
    results = Parallel(n_jobs=1)(delayed(can_converge)(smi, "test") for smi in input_smiles)
    results = pd.DataFrame(results, columns=["frag_smiles", "nconfs", "nconv_confs", "enthalpy", "converged"])

    results.to_csv(outname, index=False)
