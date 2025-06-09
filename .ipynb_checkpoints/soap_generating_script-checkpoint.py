from ase.io import read
import pandas as pd
from columns import columns
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
        # r_cut=8.0,
        n_max= n_max,
        l_max= l_max,
        # sigma = 0.2,
        average = 'inner',
        sparse=False
    )
    return soap

def compute_soap(item):
    filename, structure = item
    descriptors = soap.create(structure, n_jobs = -1)
    return filename, descriptors

def combine_lists(dict_of_lists, operation='union'):
    values = list(dict_of_lists.values())
    i = 0
    if not values:
        return []

    if operation == 'union':
        result = set()
        for lst in values:
            result.update(lst)
        return list(result)

    elif operation == 'intersection':
        result = set(values[0])
        for lst in values[1:]:
            result.intersection_update(lst)
            print(i, result)
            i += 1
        return list(result)

    else:
        raise ValueError("Operation must be 'union' or 'intersection'")

folder_path = './CIF_files'
filenames = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

mof_structures = {}
species = {}

with Pool() as pool:
    for filename, structure in tqdm(pool.imap_unordered(read_cif, filenames), total=len(filenames), desc="Reading CIFs"):
        mof_structures[filename] = structure
        species[filename] = sorted(set(structure.get_chemical_symbols()))

mof_items = list(mof_structures.items())
union_species = combine_lists(species)
n_max = 1
l_max = 1

soap_df = pd.DataFrame(columns = columns(union_species, l_max, n_max))
soap = S(union_species)
    



with Pool() as pool:
    for filename, descriptors in tqdm(pool.imap_unordered(compute_soap, mof_items), total=len(mof_items), desc="Creating SOAP descriptors"):
        df = pd.DataFrame(descriptors)
        # df["filename"] = filename  # Optional: track origin
        soap_df = pd.concat([soap_df, df], ignore_index=True)

soap_df.to_csv('averaged_local_soap_mofs.csv', index=False)  # `index=False` to avoid writing row numbers
