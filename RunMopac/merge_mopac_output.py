import glob
import pandas as pd


dfs = []
for _file in glob.glob('*_output.csv'):
    dfs.append(pd.read_csv(_file))

merged_output = pd.concat(dfs, axis=0)
merged_output.to_csv('mopac_reactions_output.csv', index=False)
