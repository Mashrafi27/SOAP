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
import pickle


def determine_all_columns(folder_path, n_max=1, l_max=1):
    """
    Step 1: Determine all possible column names by processing all MOFs
    """
    filenames = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                 if os.path.isfile(os.path.join(folder_path, f))]
    
    all_columns = set()
    
    print("Step 1: Determining all possible columns...")
    
    with Pool() as pool:
        for filename, structure, species in tqdm(pool.imap_unordered(read_cif, filenames), 
                                                total=len(filenames), 
                                                desc="Processing MOFs for columns"):
            # Create SOAP descriptor for this structure
            soap_out, soap = S(structure, species, n_max=n_max, l_max=l_max)
            
            # Get column names for this structure
            columns_this_mof = slice_column(soap, list(species))
            all_columns.update(columns_this_mof)
    
    # Convert to sorted list for consistent ordering
    all_columns = sorted(list(all_columns))
    
    print(f"Total unique columns found: {len(all_columns)}")
    
    # Save column names for reference
    with open('all_soap_columns.txt', 'w') as f:
        for col in all_columns:
            f.write(f"{col}\n")
    
    return all_columns


def process_single_mof_2d(args):
    """
    Process a single MOF and return its 2D SOAP matrix with standardized columns
    """
    filename, all_columns, n_max, l_max = args
    
    try:
        # Read structure
        structure = read(filename)
        species = set(structure.get_chemical_symbols())
        
        # Generate SOAP descriptors (2D matrix)
        soap_out, soap = S(structure, species, n_max=n_max, l_max=l_max)
        
        # Get column names for this structure
        columns_this_mof = slice_column(soap, list(species))
        
        # Create a more efficient approach to handle missing columns
        n_atoms = soap_out.shape[0]
        n_total_cols = len(all_columns)
        
        # Initialize full matrix with zeros
        full_matrix = np.zeros((n_atoms, n_total_cols))
        
        # Create column mapping
        col_mapping = {col: idx for idx, col in enumerate(all_columns)}
        
        # Fill in the existing SOAP values
        for i, col in enumerate(columns_this_mof):
            if col in col_mapping:
                full_matrix[:, col_mapping[col]] = soap_out[:, i]
        
        # Add metadata
        base_filename = os.path.basename(filename)
        
        return base_filename, full_matrix, full_matrix.shape
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None, None, None


def generate_2d_soap_matrices(folder_path, all_columns, n_max=1, l_max=1, save_format='npz'):
    """
    Step 2-4: Generate 2D SOAP matrices for all MOFs with standardized columns
    """
    filenames = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                 if os.path.isfile(os.path.join(folder_path, f))]
    
    print(f"Step 2-4: Generating 2D SOAP matrices for {len(filenames)} MOFs...")
    
    # Prepare arguments for multiprocessing
    args_list = [(filename, all_columns, n_max, l_max) for filename in filenames]
    
    soap_matrices = {}
    mof_shapes = {}
    
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_single_mof_2d, args_list), 
                           total=len(filenames), 
                           desc="Generating 2D SOAP matrices"))
    
    # Process results
    for base_filename, soap_matrix, shape in results:
        if base_filename is not None:
            soap_matrices[base_filename] = soap_matrix
            mof_shapes[base_filename] = shape
    
    print(f"Successfully processed {len(soap_matrices)} MOFs")
    print(f"Each MOF has {len(all_columns)} columns")
    
    # Save results
    if save_format == 'npz':
        # Save as compressed numpy arrays
        np.savez_compressed('soap_2d_matrices.npz', **soap_matrices)
        
        # Save metadata
        metadata = {
            'mof_names': list(soap_matrices.keys()),
            'column_names': all_columns,
            'shapes': mof_shapes,
            'n_mofs': len(soap_matrices),
            'n_columns': len(all_columns)
        }
        
        with open('soap_2d_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
            
    elif save_format == 'hdf5':
        # Alternative: Save as HDF5 for easier access
        import h5py
        
        with h5py.File('soap_2d_matrices.h5', 'w') as f:
            for mof_name, matrix in soap_matrices.items():
                f.create_dataset(mof_name, data=matrix, compression='gzip')
            
            # Save column names as attributes
            f.attrs['column_names'] = [col.encode('utf-8') for col in all_columns]
            f.attrs['n_columns'] = len(all_columns)
            f.attrs['n_mofs'] = len(soap_matrices)
    
    return soap_matrices, all_columns


def load_and_verify_results(load_format='npz'):
    """
    Utility function to load and verify the generated results
    """
    if load_format == 'npz':
        # Load matrices
        data = np.load('soap_2d_matrices.npz')
        
        # Load metadata
        with open('soap_2d_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        print(f"Loaded {metadata['n_mofs']} MOFs")
        print(f"Each MOF has {metadata['n_columns']} columns")
        
        # Verify shapes
        for mof_name in metadata['mof_names'][:5]:  # Check first 5
            matrix = data[mof_name]
            print(f"{mof_name}: shape {matrix.shape}")
        
        return data, metadata
        
    elif load_format == 'hdf5':
        import h5py
        
        with h5py.File('soap_2d_matrices.h5', 'r') as f:
            print(f"Loaded {f.attrs['n_mofs']} MOFs")
            print(f"Each MOF has {f.attrs['n_columns']} columns")
            
            # Check first few MOFs
            mof_names = list(f.keys())[:5]
            for mof_name in mof_names:
                matrix = f[mof_name][:]
                print(f"{mof_name}: shape {matrix.shape}")