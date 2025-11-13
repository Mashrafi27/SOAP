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

def read_cif(file_path):
    try:
        structure = read(file_path)
        filename = os.path.basename(file_path)
        return filename, structure
    except Exception as e:
        filename = os.path.basename(file_path)
        return filename, f"Error: {e}"

def S(species):
    soap = SOAP(
        species=species,
        periodic=False,
        r_cut=5.0,
        n_max= 1,
        l_max= 1,
        # sigma = 0.2,
        average = 'outer',
        sparse=False
    )
    return soap

def compute_soap(item):
    filename, structure = item
    descriptors = soap.create(structure, n_jobs = -1)
    return filename, descriptors


folder_path = './comb_CIF_files'
filenames = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

mof_structures = {}
species = {}

n_max = 1
l_max = 1

soap_df = pd.DataFrame()
    

with Pool() as pool:
    for filename, structure in tqdm(pool.imap_unordered(read_cif, filenames), total=len(filenames), desc="Reading CIFs"):
        mof_structures[filename] = structure
        species[filename] = set(structure.get_chemical_symbols())
        soap = S(species[filename])  
        df = pd.DataFrame([soap.create(mof_structures[filename]).flatten()], columns = slice_column(soap, list(species[filename])))
        df['filename'] = filename
        df = df.reindex(columns=soap_df.columns.union(df.columns, sort=False), fill_value=0)
        soap_df = soap_df.reindex(columns=df.columns.union(soap_df.columns, sort=False), fill_value=0)
        soap_df = pd.concat([soap_df, df], ignore_index=True)

soap_df.to_csv('outer_averaged_local_soap_mofs_6k.csv', index=False)  # `index=False` to avoid writing row numbers
