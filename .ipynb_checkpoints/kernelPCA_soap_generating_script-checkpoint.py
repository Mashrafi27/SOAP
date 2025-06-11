from ase.io import read
import pandas as pd
from columns import slice_column, columns
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from dscribe.descriptors import SOAP
import os
from tqdm import tqdm
from multiprocessing import Pool
from soap import *


folder_path = './CIF_files'
filenames = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

mof_structures = {}
species = {}

n_max = 1
l_max = 1

soap_df = pd.DataFrame()
    

with Pool() as pool:
    for filename, structure, species in tqdm(pool.imap_unordered(read_cif, filenames), total=len(filenames), desc="Reading CIFs"):
        # mof_structures[filename] = structure
        # species[filename] = sorted(set(structure.get_chemical_symbols()))
        soap_out, soap = S(structure, species)  
        kpca_out = kernelPCA(soap_out)
        df = pd.DataFrame([kpca_out], columns = slice_column(list(species), n_max, l_max))
        df['filename'] = filename
        df = df.reindex(columns=soap_df.columns.union(df.columns, sort=False), fill_value=0)
        soap_df = soap_df.reindex(columns=df.columns.union(soap_df.columns, sort=False), fill_value=0)
        soap_df = pd.concat([soap_df, df], ignore_index=True)

soap_df.to_csv('kernelPCA_soap_mofs.csv', index=False)  # `index=False` to avoid writing row numbers
