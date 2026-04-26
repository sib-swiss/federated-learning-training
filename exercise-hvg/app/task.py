"""HighVariance: A Flower for high-variance genes identification."""

import numpy as np
import pandas as pd
import random
import anndata
from typing import List, Dict


### Functions for hvg identification

def compute_hvg_client_stats(adata: anndata.AnnData, all_genes_list: List[str]) -> Dict:
    """
    Client-side computation: Compute sufficient statistics for HVG calculation.
    
    Args:
        adata: AnnData object with log-normalized expression data
        all_genes_list: List of all genes in the desired order
        
    Returns:
        Dict with 'sum', 'sum_sq', 'n_cells', 'gene_names' to send to server
    """
    # Check that all genes in all_genes_list are in adata.var_names
    missing = set(all_genes_list) - set(adata.var_names)
    if missing:
        raise ValueError(f"adata is missing genes: {missing}")
    
    # Get indices of all_genes_list in adata.var_names
    indices = [adata.var_names.get_loc(g) for g in all_genes_list]
    
    # Select columns in that order
    X = adata.X[:, indices]
    
    if hasattr(X, 'toarray'):  # Handle sparse matrices
        X = X.toarray()
    X = X.copy()
    
    # Apply expm1 transformation (reverse of log1p normalization)
    base = adata.uns.get("log1p", {}).get("base")
    if base is not None:
        X *= np.log(base)
    X = np.expm1(X)
    
    # Compute necessary statistics per gene (column-wise)
    gene_sum = np.sum(X, axis=0)         # Sum of each gene
    gene_sum_sq = np.sum(X**2, axis=0)   # Sum of squares of each gene
    n_cells = float(X.shape[0])          # Number of cells
    
    return {
        'sum': np.asarray(gene_sum).flatten(),
        'sum_sq': np.asarray(gene_sum_sq).flatten(),
        'gene_names': all_genes_list,
        'n_cells': n_cells
    }

def compute_hvg_from_stats(
    gene_sum: np.ndarray,
    gene_sum_sq: np.ndarray,
    n_cells: int,
    gene_names: List[str],
    n_top_genes: int = 2000,
    n_bins: int = 20,
    flavor: str = "seurat"
) -> pd.DataFrame:
    """
    Compute HVG selection from pre-computed mean and variance.
    This replicates the logic from scanpy.pp.highly_variable_genes.
    
    Args:
        gene_sum: Sum of each gene
        gene_sum_sq: Sum of squares of each gene
        n_cells: Number of cells
        gene_names: List of gene names
        n_top_genes: Number of top genes to select
        n_bins: Number of bins for mean expression
        flavor: HVG flavor ('seurat' or 'cell_ranger')
    
    Returns:
        DataFrame with HVG annotations
    """
    n_genes = len(gene_sum)
    
    # Compute gene mean
    gene_mean = gene_sum / n_cells
    
    # Compute global variance: Var(X) = E[X^2] - E[X]^2 with Bessel's correction
    global_var = (gene_sum_sq - n_cells * gene_mean**2) / (n_cells - 1)
    global_var = np.maximum(global_var, 0)  # Ensure no negative variances

    # Compute dispersion = var / mean
    mean = gene_mean.copy()
    mean[mean == 0] = 1e-12  # Avoid division by zero
    dispersion = global_var / mean
    
    if flavor == "seurat":
        dispersion[dispersion == 0] = np.nan
        dispersion = np.log(dispersion)
        mean = np.log1p(mean)
    
    # Create DataFrame
    df = pd.DataFrame({
        'means': mean,
        'dispersions': dispersion
    }, index=gene_names)
    
    # Bin genes by mean expression
    if flavor == "seurat":
        df['mean_bin'] = pd.cut(df['means'], bins=n_bins)
    elif flavor == "cell_ranger":
        bins = np.r_[-np.inf, np.percentile(df['means'], np.arange(10, 105, 5)), np.inf]
        df['mean_bin'] = pd.cut(df['means'], bins=bins)
    
    # Compute normalized dispersion within each bin
    disp_grouped = df.groupby('mean_bin', observed=True)['dispersions']
    
    if flavor == "seurat":
        disp_bin_stats = disp_grouped.agg(avg='mean', dev='std')
        
        # Handle single-gene bins: set normalized dispersion to 1
        one_gene_per_bin = disp_bin_stats['dev'].isnull()
        disp_bin_stats.loc[one_gene_per_bin, 'dev'] = disp_bin_stats.loc[one_gene_per_bin, 'avg']
        disp_bin_stats.loc[one_gene_per_bin, 'avg'] = 0
        
    elif flavor == "cell_ranger":
        from statsmodels.robust import mad
        disp_bin_stats = disp_grouped.agg(avg='median', dev=mad)
    
    # Map bin stats back to genes
    disp_stats = disp_bin_stats.loc[df['mean_bin']].set_index(df.index)
    
    # Normalize dispersion
    df['dispersions_norm'] = (df['dispersions'] - disp_stats['avg']) / disp_stats['dev']
    
    # Select top genes by normalized dispersion
    dispersion_norm = df['dispersions_norm'].values.copy()
    dispersion_norm = np.nan_to_num(dispersion_norm, nan=-np.inf)
    
    if n_top_genes is not None and n_top_genes < len(gene_names):
        sorted_disp = np.sort(dispersion_norm)[::-1]
        threshold = sorted_disp[n_top_genes - 1]
        df['highly_variable'] = dispersion_norm >= threshold
    else:
        df['highly_variable'] = True
    
    return df

