"""Utility functions for FedSCVI: a Flower app for federated single-cell variational inference.

This file groups the reusable helper functions used by the Flower ServerApp
and ClientApp.

The utilities are organized as follows:

1. File I/O helpers
2. Data loading helpers
3. Model weight helpers
4. SCVI model construction helpers
5. Evaluation helpers
6. Model saving helpers
"""

from pathlib import Path
from typing import Any, Union

import json
import os

import anndata
import numpy as np
import pandas as pd
import scvi
import torch
from flwr.app import ArrayRecord


# ============================================================
# 1. File I/O helpers
# ============================================================


def read_json(path: Union[str, Path]) -> Any:
    """
    Read a JSON file and return its parsed contents.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the JSON file.

    Returns
    -------
    Any
        Parsed JSON content. The returned object depends on the file contents,
        for example a list, dictionary, string, or number.
    """
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# 2. Data loading helpers
# ============================================================


def load_local_data_simulation(
    partition_id: int,
    data_file_path: str,
) -> anndata.AnnData:
    """
    Load a partition of the gene expression AnnData corresponding to a single technology.

    This helper is useful in simulation settings where one global AnnData file
    is split into several clients. Each client receives all cells belonging to
    one value of `adata.obs["tech"]`.

    Parameters
    ----------
    partition_id : int
        Index of the technology to select, based on the sorted unique values in
        `adata.obs["tech"]`.
    data_file_path : str
        Path to the `.h5ad` file.

    Returns
    -------
    anndata.AnnData
        The selected AnnData partition containing cells from one technology.
    """
    adata = anndata.read_h5ad(data_file_path)

    techs = sorted(map(str, adata.obs["tech"].unique()))
    selected_tech = techs[partition_id]

    partition = adata[adata.obs["tech"].astype(str) == selected_tech].copy()
    return partition


def create_dummy_adata_from_hvg(
    hvg_list: list[str],
    batch_list: list[str],
    num_cells: int = 100,
    fill_value: float = 0.0,
) -> anndata.AnnData:
    """
    Create a minimal AnnData object with the same gene and batch structure as the clients.

    The server does not own real training data. However, it still needs to
    instantiate an SCVI model with the same number of genes and the same batch
    categories as the client models. This dummy AnnData is used only for model
    initialization on the server.

    Parameters
    ----------
    hvg_list : list of str
        List of highly variable genes used as model input features.
    batch_list : list of str
        List of all possible batch/technology labels.
    num_cells : int, optional
        Number of dummy cells to create. Defaults to 100.
    fill_value : float, optional
        Value used to fill the dummy count matrix. Defaults to 0.0.

    Returns
    -------
    anndata.AnnData
        Dummy AnnData object with:
        - `X` containing a dense dummy matrix,
        - `layers["counts"]` containing the same matrix,
        - `obs["tech"]` containing categorical batch labels,
        - `var_names` matching `hvg_list`.
    """
    n_genes = len(hvg_list)

    X = np.full(
        shape=(num_cells, n_genes),
        fill_value=fill_value,
        dtype=np.float32,
    )

    obs = pd.DataFrame(
        {
            "tech": pd.Categorical(
                [batch_list[i % len(batch_list)] for i in range(num_cells)],
                categories=batch_list,
            )
        },
        index=pd.Index([f"cell_{i}" for i in range(num_cells)], dtype="object"),
    )

    var = pd.DataFrame(
        index=pd.Index(hvg_list, dtype="object"),
    )

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.layers["counts"] = adata.X.copy()
    adata.obs["tech"] = pd.Categorical(adata.obs["tech"], categories=batch_list)

    return adata


# ============================================================
# 3. Model weight helpers
# ============================================================


def get_weights(model) -> list[np.ndarray]:
    """
    Extract model weights as a list of NumPy arrays.

    Flower exchanges model parameters as NumPy arrays. SCVI models wrap the
    underlying PyTorch module in `model.module`, so we read the PyTorch
    `state_dict()` from there.

    Parameters
    ----------
    model : scvi.model.SCVI
        SCVI model whose parameters should be extracted.

    Returns
    -------
    list of numpy.ndarray
        Model parameters converted from PyTorch tensors to NumPy arrays.
    """
    return [v.detach().cpu().numpy() for v in model.module.state_dict().values()]


def set_weights(model, weights: list[np.ndarray]) -> None:
    """
    Load a list of NumPy arrays into an SCVI model.

    The order of `weights` must match the order returned by `get_weights`.
    `strict=True` is used so that shape mismatches are caught immediately.

    Parameters
    ----------
    model : scvi.model.SCVI
        SCVI model whose parameters should be updated.
    weights : list of numpy.ndarray
        Model parameters received from Flower.

    Returns
    -------
    None
    """
    state_dict = model.module.state_dict()

    new_state_dict = {
        key: torch.tensor(weight)
        for key, weight in zip(state_dict.keys(), weights)
    }

    model.module.load_state_dict(new_state_dict, strict=True)


# ============================================================
# 4. SCVI model construction helpers
# ============================================================


def get_architecture(cfg: dict | None = None) -> dict:
    """
    Read SCVI architecture settings from the Flower run config.

    This function keeps architecture choices in one place, making sure that the
    server and all clients instantiate compatible SCVI models.

    Parameters
    ----------
    cfg : dict, optional
        Flower run configuration. If `None`, default SCVI architecture settings
        are used.

    Returns
    -------
    dict
        Keyword arguments passed to `scvi.model.SCVI`.
    """
    cfg = cfg or {}

    return {
        "use_layer_norm": cfg.get("use_layer_norm", "both"),
        "use_batch_norm": cfg.get("use_batch_norm", "none"),
        "encode_covariates": bool(cfg.get("encode_covariates", True)),
        "dropout_rate": float(cfg.get("dropout_rate", 0.2)),
        "n_layers": int(cfg.get("n_layers", 2)),
    }


def create_scvi_model(
    adata: anndata.AnnData,
    batch_list: list[str],
    arch_cfg: dict | None = None,
):
    """
    Construct and return an SCVI model.

    `setup_anndata` tells scvi-tools how to interpret the AnnData object:
    - the raw/count data are stored in `adata.layers["counts"]`,
    - the batch variable is stored in `adata.obs["tech"]`.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object used to initialize the SCVI model.
    batch_list : list of str
        List of all expected technology/batch categories.
    arch_cfg : dict, optional
        Architecture keyword arguments. If `None`, defaults from
        `get_architecture` are used.

    Returns
    -------
    scvi.model.SCVI
        Initialized SCVI model.
    """
    adata.obs["tech"] = pd.Categorical(
        adata.obs["tech"].astype(str),
        categories=batch_list,
    )

    scvi.model.SCVI.setup_anndata(
        adata,
        batch_key="tech",
        layer="counts",
    )

    arch = get_architecture(arch_cfg)
    return scvi.model.SCVI(adata, **arch)


# ============================================================
# 5. Evaluation helpers
# ============================================================


def get_loss(model, adata: anndata.AnnData) -> float:
    """
    Compute the loss used for reporting client evaluation metrics.

    SCVI reports the evidence lower bound, or ELBO. Since training losses are
    conventionally minimized, we return the negative ELBO as a loss value.

    Parameters
    ----------
    model : scvi.model.SCVI
        Trained or partially trained SCVI model.
    adata : anndata.AnnData
        AnnData object on which to evaluate the model.

    Returns
    -------
    float
        Negative ELBO value.
    """
    elbo = model.get_elbo(adata)
    return -float(elbo)


# ============================================================
# 6. Model saving helpers
# ============================================================


def save_final_model(model, path: str = "./model.pt") -> None:
    """
    Save the final SCVI model to disk.

    Parameters
    ----------
    model : scvi.model.SCVI
        SCVI model to save.
    path : str, optional
        Output directory or path used by `model.save`. Defaults to "./model.pt".

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save(path, overwrite=True, save_anndata=True)
    print(f"[INFO] Saved final model to {path}")


def make_on_final_arrays(model, set_weights, save_path: str = "./model.pt"):
    """
    Create a callback that saves the final aggregated Flower arrays as an SCVI model.

    Custom server strategies receive the final global model as a Flower
    `ArrayRecord`. This helper converts that `ArrayRecord` back to NumPy arrays,
    loads them into the SCVI model, and then saves the model to disk.

    Parameters
    ----------
    model : scvi.model.SCVI
        Server-side SCVI model object used as a container for final weights.
    set_weights : callable
        Function used to load a list of NumPy arrays into `model`.
    save_path : str, optional
        Path where the final model should be saved. Defaults to "./model.pt".

    Returns
    -------
    callable
        Callback function with signature `callback(final_arrays: ArrayRecord)`.
    """

    def _on_final_arrays(final_arrays: ArrayRecord) -> None:
        """
        Apply final aggregated arrays to the model and save it.

        Parameters
        ----------
        final_arrays : flwr.app.ArrayRecord
            Aggregated model parameters produced by the server strategy.

        Returns
        -------
        None
        """
        final_weights = final_arrays.to_numpy_ndarrays()
        set_weights(model, final_weights)

        if hasattr(model, "is_trained"):
            try:
                model.is_trained = True
            except Exception:
                pass

        save_final_model(model, save_path)

    return _on_final_arrays