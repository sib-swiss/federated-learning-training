"""HighVariance: A Flower for high-variance genes identification."""

import numpy as np
import pandas as pd
import random
import anndata

### Functions for the exercise starting point

def get_dummy_start():
    """Return a dummy initial parameter array."""
    return np.ones((1, 1))

def get_random_integer(N=5):
    """Return a random integer between 1 and N (inclusive)."""
    return random.randint(1, N)

### Provide a single technology to each client

def load_local_data(partition_id: int, data_file_path: str):
    """
    Load a partition of the gene expression anndata object corresponding to a single technology.

    Parameters:
        partition_id (int): The index of the technology to select.
        data_file_path (str): Path to the .h5ad file.

    Returns:
        anndata.AnnData: The selected partition of the dataset.
    """
    adata = anndata.read_h5ad(data_file_path)
    techs = adata.obs['tech'].unique()
    partition = adata[adata.obs['tech'] == techs[partition_id]]
    return partition