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
    structure = read(file_path)
    filename = os.path.basename(file_path)
    species = set(structure.get_chemical_symbols())
    return filename, structure, species


def S(structure, species, r_cut = 5.0, n_max = 1, l_max = 1, sigma = 1.0, average = "off", periodic = "False"):
    soap = SOAP(
        species=species,
        periodic=False,
        r_cut=r_cut,
        n_max= n_max,
        l_max= l_max,
        sigma = sigma,
        average = average,
        sparse=False
    )
    output = soap.create(structure)
    return output, soap

def compute_soap(item):
    filename, structure = item
    descriptors = soap.create(structure, n_jobs = -1)
    return filename, descriptors

def kernelPCA(soap_out, Tr = False):

    def rbf_kernel(u, v, gamma=1e-3):
        diff = u - v
        return np.exp(-gamma * (diff @ diff))
    
    N_env, N_feat = soap_out.shape
    if Tr:
        N = N_feat
    else:
        N = N_env
    
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if Tr:
                K[i, j] = rbf_kernel(soap_out[:,i], soap_out[:,j], gamma=1e-3)
            else:    
                K[i, j] = rbf_kernel(soap_out[i], soap_out[j], gamma=1e-3)
    
    eigvals, eigvecs = np.linalg.eigh(K)
    alpha = eigvecs[:, -1]       # shape = (N_env,)
    # (Optionally, you can normalize alpha so that sum(alpha^2)=1 or so that sum(alpha)=1, etc.)
    alpha = alpha / np.linalg.norm(alpha)
    
    # --- 3) Form the weighted‐sum row d = alpha^T @ soap_out ---
    

    if Tr:
        d = (soap_out * alpha).mean(axis=0, keepdims=True)
    else:
        d = alpha.reshape(1, N_env) @ soap_out   # shape = (1, N_feat)

    return d[0]