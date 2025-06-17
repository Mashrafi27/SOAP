import os
from multiprocessing import Pool
from tqdm import tqdm
from ..soap import read_cif, S

def _process_cif(args):
    """Load a CIF file, optionally truncate, and compute its SOAP descriptor using user-defined S."""
    path, soap_params, max_envs = args
    filename, structure, species = read_cif(path)
    if max_envs is not None and len(structure) > max_envs:
        structure = structure[:max_envs]
    soap_out, _ = S(structure, species, **soap_params)
    return soap_out, filename


def load_structures_and_compute_soaps(cif_dir, soap_params, max_envs=None, n_workers=4):
    """
    Load all CIF files in `cif_dir` and compute SOAP descriptors in parallel
    using the user-provided `S` function from soap.py.

    Args:
        cif_dir (str): directory containing .cif files
        soap_params (dict): parameters for S (r_cut, n_max, l_max, sigma, average)
        max_envs (int, optional): max number of atoms/environments per structure
        n_workers (int): number of parallel workers

    Returns:
        soap_data (list of np.ndarray): list of [K_i, D] SOAP matrices
        cif_names (list of str): corresponding CIF filenames
    """
    # Gather CIF file paths
    cif_paths = [os.path.join(cif_dir, f) for f in os.listdir(cif_dir) if f.endswith('.cif')]

    # Prepare arguments for parallel processing
    args_list = [(path, soap_params, max_envs) for path in cif_paths]

    soap_data = []
    cif_names = []
    # Parallel SOAP computation
    with Pool(processes=n_workers) as pool:
        for mat, name in tqdm(pool.imap_unordered(_process_cif, args_list), total=len(args_list), desc='Computing SOAPs'):
            soap_data.append(mat)
            cif_names.append(name)
    return soap_data, cif_names