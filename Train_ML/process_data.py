import pandas as pd
import numpy as np

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

print(rdkit.__version__)


def make_molobj(smi):
    """ """
    molobj = Chem.MolFromSmiles(smi, sanitize=False)
    Chem.SanitizeMol(molobj)
    return molobj


def ECFP(mol, radius=2, length=1024):
    """ """
    arr = np.empty((0,),dtype=np.int8)
    bi = {}
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=length, bitInfo=bi)
    DataStructs.ConvertToNumpyArray(fp1,arr)
    return arr, bi


def convergence_criteria(df, cutoff=1/3):
    """ If > cutoff conformers converged the fragment is considered to converge."""
    new_conv = []
    for frag in df.itertuples(index=False):
        if  frag.nconfs == 0: # Can't make conformer
            conv = False
        elif frag.nconfs == frag.nconv_confs: # Perfekt conf
            conv = True
        elif frag.nconfs * cutoff <= frag.nconv_confs: # decent
            conv = True
        else: # rest
            conv = False
    new_conv.append(conv)
    df['target'] = new_conv
    return df 


def process_data(smote=False, n_jobs=5):
    mopac_output_data = pd.read_csv('mopac_fragments_output.csv')
    mopac_output_data['molobj'] = mopac_output_data['frag_smiles'].apply(make_molobj)
    mopac_output_data['tmp'] = mopac_output_data['molobj'].apply(ECFP)

    # Split tuple
    tmp_df = pd.DataFrame(mopac_output_data.tmp.tolist(), columns=['ecfp_bits', 'bit_info'])
    mopac_output_data = pd.concat([mopac_output_data, tmp_df], axis=1)
    mopac_output_data.drop(columns='tmp', inplace=True)

    mopac_output_data = convergence_criteria(mopac_output_data)

    mopac_output_data.to_pickle('processed_data/fragments_data.pkl')

    # Make training data
    X = np.stack(mopac_output_data['ecfp_bits']).astype(np.int8)
    y = mopac_output_data['new_conv'].to_numpy().astype(np.int8)

    with open("processed_data/xtrain.npy") as fxtrain:
        np.save(fxtrain, X)
    
    with open("processed_data/ytrain.npy") as fytrain:
        np.save(fytrain, y)

    if smote:
        from imblearn.over_sampling import SMOTE 
        sm = SMOTE(random_state=31, n_jobs=n_jobs)
        Xres, yres = sm(X, y)
        with open("processed_data/xtrain_smote.npy", "wb") as fxtrain_smote:
            np.save(fxtrain_smote, Xres).astype(np.int8)
        with open("processed_data/train_smote.npy", "wb") as fytrain_smote:
            np.save(fytrain_smote, yres).astype(np.int8)
