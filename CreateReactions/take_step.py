import itertools
import copy
import numpy as np

from functools import partial

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdchem
from rdkit.Chem.rdchem import ResonanceFlags
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from rdkit.Chem.EnumerateStereoisomers import StereoEnumerationOptions

from numba import jit
from numba.typed import Dict
from numba import types

from xyz2mol import get_proto_mol, AC2mol


n = 0
bad_mols = []

key_type = types.int64
val_type = types.int32[:]


@jit(nopython=True)
def valid_product_numba(adj_matrix, atomic_num):
    """Check that the produces product is valid according to the valence"""

    max_valence_numba = Dict.empty(key_type=key_type, value_type=val_type)

    max_valence_numba[1] = np.array([0, 1], dtype=np.int32)
    max_valence_numba[5] = np.array([0, 3], dtype=np.int32)
    max_valence_numba[6] = np.array([0, 4], dtype=np.int32)
    max_valence_numba[7] = np.array([0, 4], dtype=np.int32)
    max_valence_numba[8] = np.array([0, 3], dtype=np.int32)
    max_valence_numba[9] = np.array([0, 1], dtype=np.int32)
    max_valence_numba[14] = np.array([0, 4], dtype=np.int32)
    max_valence_numba[15] = np.array([0, 5], dtype=np.int32)
    max_valence_numba[16] = np.array([0, 6], dtype=np.int32)
    max_valence_numba[17] = np.array([0, 1], dtype=np.int32)
    max_valence_numba[35] = np.array([0, 1], dtype=np.int32)
    max_valence_numba[53] = np.array([0, 1], dtype=np.int32)

    natoms = adj_matrix.shape[0]
    product_valence = np.empty(natoms, dtype=np.int64)
    for i in range(natoms):
        valence = 0
        for conection in adj_matrix[i]:
            valence += conection
        product_valence[i] = valence

    for atom, valence in zip(atomic_num, product_valence):
        if (
            valence > max_valence_numba[atom][1]
            or valence == max_valence_numba[atom][0]
        ):
            return False
    return True


def reassign_atom_idx(mol):
    """Assigns RDKit mol atom id to atom mapped id"""
    renumber = [(atom.GetIdx(), atom.GetAtomMapNum()) for atom in mol.GetAtoms()]
    new_idx = [idx[0] for idx in sorted(renumber, key=lambda x: x[1])]
    mol = Chem.RenumberAtoms(mol, new_idx)
    Chem.rdmolops.AssignStereochemistry(mol, force=True)
    return mol


def most_rigid_resonance(mol):
    """Return the most rigid resonance structure"""
    all_resonance_structures = [
        res
        for res in rdchem.ResonanceMolSupplier(mol, ResonanceFlags.UNCONSTRAINED_ANIONS)
    ]

    min_rot_bonds = 9999
    most_rigid_res = copy.deepcopy(mol)
    if len(all_resonance_structures) <= 1:  # 0 is kind weird
        return most_rigid_res

    for res in all_resonance_structures:
        Chem.SanitizeMol(res)
        num_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(res)
        if num_rot_bonds < min_rot_bonds:
            most_rigid_res = copy.deepcopy(res)
            min_rot_bonds = num_rot_bonds

    return most_rigid_res


def get_isomers(reactant, product):
    """Produce all combinations of isomers (R/S and cis/trans). But force
    product atoms with unchanged neighbors to the same label chirality as
    the reactant"""

    product = reassign_atom_idx(product)
    reactant = reassign_atom_idx(reactant)

    # Find chiral atoms - including label chirality
    chiral_atoms_product = Chem.FindMolChiralCenters(product, includeUnassigned=True)

    unchanged_atoms = []
    for atom, _ in chiral_atoms_product:
        product_neighbors = [
            a.GetIdx() for a in product.GetAtomWithIdx(atom).GetNeighbors()
        ]
        reactant_neighbors = [
            a.GetIdx() for a in reactant.GetAtomWithIdx(atom).GetNeighbors()
        ]

        if sorted(product_neighbors) == sorted(reactant_neighbors):
            unchanged_atoms.append(atom)

    # make combinations of isomers.
    opts = StereoEnumerationOptions(onlyUnassigned=False, unique=False)
    Chem.rdmolops.AssignStereochemistry(
        product, cleanIt=True, flagPossibleStereoCenters=True, force=True
    )

    product_isomers_mols = []
    product_isomer_smiles = []
    for product_isomer in EnumerateStereoisomers(product, options=opts):
        Chem.rdmolops.AssignStereochemistry(product_isomer, force=True)
        for atom in unchanged_atoms:
            reactant_global_tag = reactant.GetAtomWithIdx(atom).GetProp("_CIPCode")
            # TODO make sure that the _CIPRank is the same for atom in reactant and product.
            try:
                product_isomer_global_tag = product_isomer.GetAtomWithIdx(atom).GetProp(
                    "_CIPCode"
                )
                if reactant_global_tag != product_isomer_global_tag:
                    product_isomer.GetAtomWithIdx(atom).InvertChirality()
            except KeyError:
                continue

        # Make product canonical
        product_isomer = Chem.MolFromSmiles(
            Chem.MolToSmiles(product_isomer), sanitize=False
        )
        isomer_smiles = Chem.MolToSmiles(product_isomer)
        if isomer_smiles not in product_isomer_smiles:
            product_isomer_smiles.append(isomer_smiles)
            product_isomers_mols.append(product_isomer)

    return product_isomers_mols


class CreateValidIs:
    def __init__(
        self, mapped_molecule, max_num_bonds: int = 2, cd: int = 4, inactive_atoms=[]
    ) -> None:

        self._max_bonds = max_num_bonds
        self._cd = cd
        self._inactive_atoms = inactive_atoms

        self._mapped_molecule = mapped_molecule
        self._ac_matrix = Chem.GetAdjacencyMatrix(self._mapped_molecule)

        self._atom_number = np.array(
            [atom.GetAtomicNum() for atom in self._mapped_molecule.GetAtoms()]
        )

    def _make_simple_conversion_matrices(self) -> None:
        """ """
        # Create one-bond conversion matrices
        num_atoms = len(self._atom_number)

        # inital inactive atoms is atom map numbers.
        # convert to 0 based idx.
        inactive_atoms = [int(x) - 1 for x in self._inactive_atoms]

        make1, break1 = [], []
        for i in range(num_atoms):
            for j in range(num_atoms):
                # Check if i or j is an inactive atoms don't add the conversion matrix.
                if (i in inactive_atoms) or (j in inactive_atoms):
                    continue

                conversion_matrix = np.zeros(self._ac_matrix.shape, np.int8)
                if j > i:
                    if self._ac_matrix[i, j] == 0:
                        conversion_matrix[i, j] = conversion_matrix[j, i] = 1
                        make1.append(conversion_matrix)
                    else:
                        conversion_matrix[i, j] = conversion_matrix[j, i] = -1
                        break1.append(conversion_matrix)

        self._make1 = make1
        self._break1 = break1

    def _make_break_combinations(self):
        """ """
        product_combination = []
        for make_break in itertools.product(range(self._max_bonds + 1), repeat=2):
            if sum(make_break) <= self._cd and max(make_break) <= self._max_bonds:
                product_combination.append(make_break)
        return product_combination[1:]  # skip 0,0 = reactant.

    def __iter__(self):
        """ """
        valid_valence_filter = partial(
            valid_product_numba, atomic_num=self._atom_number
        )

        self._make_simple_conversion_matrices()

        for num_make, num_break in self._make_break_combinations():
            make_combs = itertools.combinations(self._make1, num_make)
            break_combs = itertools.combinations(self._break1, num_break)

            for conv_matrix in itertools.product(make_combs, break_combs):
                conv_matrix = np.array(sum(conv_matrix, ())).sum(axis=0)
                product_ac_matrix = self._ac_matrix + conv_matrix

                if valid_valence_filter(product_ac_matrix):
                    yield product_ac_matrix


class TakeElementaryStep:
    def __init__(
        self, mapped_molecule, max_num_bonds: int = 3, cd: int = 5, inactive_atoms=[]
    ) -> None:

        self._max_bonds = max_num_bonds
        self._cd = cd
        self._inactive_atoms = inactive_atoms

        self._mapped_molecule = mapped_molecule
        self._atom_number = [
            atom.GetAtomicNum() for atom in self._mapped_molecule.GetAtoms()
        ]

    def _ac_to_mapped_products(self, product_ac_matrix):
        """ """
        proto_mol = get_proto_mol(self._atom_number)
        mol = AC2mol(
            proto_mol,
            product_ac_matrix,
            self._atom_number,
            charge=Chem.GetFormalCharge(self._mapped_molecule),
            allow_charged_fragments=True,
            use_graph=True,
            use_atom_maps=True,
        )

        if mol is None:
            return []

        Chem.SanitizeMol(mol)
        mol = most_rigid_resonance(mol)

        return get_isomers(self._mapped_molecule, mol)

    def get_products(self):

        valid_Is = CreateValidIs(
            self._mapped_molecule,
            max_num_bonds=self._max_bonds,
            cd=self._cd,
            inactive_atoms=self._inactive_atoms,
        )

        mapped_products = []
        for prod_ac in valid_Is:
            mapped_products.append(self._ac_to_mapped_products(prod_ac))

        mapped_products = list(itertools.chain.from_iterable(mapped_products))

        return mapped_products
