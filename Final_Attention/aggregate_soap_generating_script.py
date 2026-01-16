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


def aggregate(encoder_decoder, property_predictor):
    folder_path = '../CIF_files'
    filenames = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Determine all 484 columns first (do this once before the loop)
    print("Determining all unique SOAP columns...")
    all_soap_columns = set()
    for filename in tqdm(filenames, desc="Sampling for columns"):  # Sample to get all columns
        try:
            _, structure, species = read_cif(filename)
            soap_out, soap = S(structure, species)
            columns = slice_column(soap, list(species))
            all_soap_columns.update(columns)
        except:
            continue

    all_soap_columns = sorted(list(all_soap_columns))
    print(f"Found {len(all_soap_columns)} unique SOAP columns")

    # Now process all files with fixed columns

    # aggregated_column_names = []
    # # First 484 columns (seed0)
    # for col in all_soap_columns:
    #     aggregated_column_names.append(f"{col}_seed0")
    # # Next 484 columns (seed1) 
    # for col in all_soap_columns:
    #     aggregated_column_names.append(f"{col}_seed1")

    # soap_df = pd.DataFrame(columns=all_soap_columns + ['filename'])
    soap_df = pd.DataFrame()


    with Pool() as pool:
        for filename, structure, species in tqdm(pool.imap_unordered(read_cif, filenames), total=len(filenames), desc="Reading CIFs"):
            try:
                soap_out, soap = S(structure, species)
                
                # Get current columns for this MOF
                current_columns = slice_column(soap, list(species))
                
                # Create padded SOAP features (484 dimensions)
                padded_soap = np.zeros((soap_out.shape[0], len(all_soap_columns)))
                
                # Fill in actual values where columns exist
                for i, col in enumerate(current_columns):
                    if col in all_soap_columns:
                        col_idx = all_soap_columns.index(col)
                        padded_soap[:, col_idx] = soap_out[:, i]
                
                # Now pass padded SOAP to encoder_decoder
                aggr_out = set_transform_aggregation(padded_soap, encoder_decoder, property_predictor)
                
                # Create DataFrame with all columns (no more reindexing!)
                df = pd.DataFrame([aggr_out])#, columns=all_soap_columns)
                df['filename'] = filename
                
                soap_df = pd.concat([soap_df, df], ignore_index=True)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    return soap_df

# soap_df.to_csv('aggregate_soap_mofs.csv', index=False)  # `index=False` to avoid writing row numbers


