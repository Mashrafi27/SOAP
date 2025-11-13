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
from torch_geometric.nn.aggr import SetTransformerAggregation
import torch
from sklearn.decomposition import PCA


def read_cif(file_path):
    structure = read(file_path)
    filename = os.path.basename(file_path)
    species = set(structure.get_chemical_symbols())
    return filename, structure, species


def S(structure, species, r_cut = 5.0, n_max = 1, l_max = 1, sigma = 1.0, average = "off"):
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

def max_pool(soap_out):
    return np.max(soap_out, axis=0)

def pca(soap_out):
    data_2d = np.array(soap_out)
    result = []
    
    # Apply PCA to each column
    for col_idx in range(data_2d.shape[1]):
        column = data_2d[:, col_idx].reshape(-1, 1)
        
        # PCA with 1 component
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(column)
        
        # Extract the single principal component value
        result.append(pca_result.flatten()[0])
    
    return np.array(result)


def set_transform_aggregation(soap_out, encoder_decoder):
    # encoder_decoder = SetTransformerAggregation(soap_out.shape[1])
    soap_tensor = torch.FloatTensor(soap_out)
    group_indices = torch.zeros(soap_tensor.shape[0], dtype=torch.long)
    output = encoder_decoder.forward(soap_tensor, index = group_indices)
    return output[0].detach().numpy()

def kernelPCA(soap_out, Tr = False, avg = False):

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

    if avg:
        return d[0]/N_env
    else:
        return d[0]