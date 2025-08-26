"""FedSCVI: A Flower for federated single cell variational inference."""

import json
import anndata
import numpy as np
import pandas as pd
import scvi
import torch
import json
from pathlib import Path
from typing import Any, Union


def read_json(path: Union[str, Path]) -> Any:
    """Read and return the parsed contents of a JSON file at `path`."""
    with open(path, "r") as f:
        return json.load(f)


def load_local_data_simulation(partition_id: int, data_file_path: str) -> anndata.AnnData:
    """
    Load a partition of the gene expression AnnData corresponding to a single technology.

    Parameters
    ----------
    partition_id : int
        Index of the technology to select (based on `adata.obs['tech'].unique()`).
    data_file_path : str
        Path to the `.h5ad` file.

    Returns
    -------
    anndata.AnnData
        The selected partition (view) of the dataset.
    """
    adata = anndata.read_h5ad(data_file_path)
    techs = sorted(map(str, adata.obs["tech"].unique()))
    partition = adata[adata.obs["tech"] == techs[partition_id]]
    return partition


def get_weights(model):
    """Extract model weights as a list of NumPy arrays from the underlying Torch module."""
    # scvi.model.SCVI wraps the torch model in `.module`
    return [v.detach().cpu().numpy() for v in model.module.state_dict().values()]


def set_weights(model, weights):
    """Load weights (list of NumPy arrays) into the underlying Torch module, preserving key order."""
    state_dict = model.module.state_dict()
    new_state_dict = {k: torch.tensor(w) for k, w in zip(state_dict.keys(), weights)}
    model.module.load_state_dict(new_state_dict, strict=True)


def create_dummy_adata_from_hvg(hvg_list, batch_list, num_cells=100, fill_value=0.0):
    """Create a minimal AnnData with genes from `hvg_list` and a 'tech' batch column."""
    n_genes = len(hvg_list)
    X = np.full((num_cells, n_genes), fill_value, dtype=np.float32)

    obs = pd.DataFrame(
        {
            "tech": pd.Categorical(
                [batch_list[i % len(batch_list)] for i in range(num_cells)],
                categories=batch_list,
            )
        },
        index=pd.Index([f"cell_{i}" for i in range(num_cells)], dtype="object"),
    )
    var = pd.DataFrame(index=pd.Index(hvg_list, dtype="object"))

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = adata.X.copy()
    adata.obs["tech"] = pd.Categorical(adata.obs["tech"], categories=batch_list)
    return adata


def get_architecture(cfg: dict | None = None) -> dict:
    """Return SCVI architecture kwargs (pure config)."""
    cfg = cfg or {}
    return {
        "use_layer_norm":   cfg.get("use_layer_norm", "both"),
        "use_batch_norm":   cfg.get("use_batch_norm", "none"),
        "encode_covariates": bool(cfg.get("encode_covariates", True)),
        "dropout_rate":     float(cfg.get("dropout_rate", 0.2)),
        "n_layers":         int(cfg.get("n_layers", 2)),
    }


def create_scvi_model(adata, batch_list, arch_cfg: dict | None = None):
    """Construct and return an SCVI model."""
    adata.obs["tech"] = pd.Categorical(adata.obs["tech"], categories=batch_list)
    scvi.model.SCVI.setup_anndata(adata, batch_key="tech", layer="counts")
    arch = get_architecture(arch_cfg)
    return scvi.model.SCVI(adata, **arch)


def get_loss(model, adata):
    # Ensure 'batch' exists in adata.obs for evaluation
    if "batch" not in adata.obs:
        adata.obs["batch"] = adata.obs["tech"] if "tech" in adata.obs else "batch0"
    # Evaluate ELBO on the given AnnData
    elbo = model.get_elbo(adata)
    return -float(elbo)


### To be used later

def load_local_data_deployment(data_file_path: str) -> anndata.AnnData:
    """
    Loads an AnnData dataset.

    Parameters
    ----------
    data_file_path : str
        Path to the `.h5ad` file.

    Returns
    -------
    anndata.AnnData
        The selected partition (view) of the dataset.
    """
    adata = anndata.read_h5ad(data_file_path)
    return adata