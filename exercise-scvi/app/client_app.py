"""FedSCVI: A Flower for federated single cell variational inference."""

# Standard library
import gc
from pathlib import Path

# Third-party libraries
import anndata
import torch

# Flower Message API imports
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

# Local helper functions used by the client
from app.task import (
    create_scvi_model,
    get_architecture,
    get_loss,
    get_weights,
    load_local_data_simulation,
    read_json,
    set_weights,
)

# Use higher precision matrix multiplications when possible
# (can slightly improve training stability/performance)
torch.set_float32_matmul_precision("high")

# Create the Flower client application
app = ClientApp()


# ---------------------------------------------------------------------
# Helper function: load local client state
# ---------------------------------------------------------------------

def _load_client_state(context: Context):
    """
    Load client-specific data and initialize the local SCVI model.

    Each Flower client corresponds to one data partition
    (for example, one sequencing technology).

    Steps:
    1. Read client ID from Flower context
    2. Load local train/validation datasets
    3. Restrict genes to the selected HVG list
    4. Build a local SCVI model
    """

    # Flower assigns each client a partition id
    client_id = context.node_config["partition-id"]

    # Build client folder path using partition id
    client_data_folder = Path(
        context.run_config["client_folder_path"].format(
            partition_id=client_id
        )
    )

    # ---------------------------------------------------------
    # Load metadata files
    # ---------------------------------------------------------

    # Top highly-variable genes used in training
    hvg_file_path = client_data_folder / context.run_config["top2k_genes_file"]
    hvg_list = read_json(hvg_file_path)

    # Full list of possible technologies / batches
    batch_file_path = client_data_folder / context.run_config["all_techs_file"]
    batch_list = read_json(batch_file_path)

    # ---------------------------------------------------------
    # Load local train / validation data
    # ---------------------------------------------------------

    adata_local_train = anndata.read_h5ad(client_data_folder / "train.h5ad")
    adata_local_train = adata_local_train[:, hvg_list].copy()

    adata_local_valid = anndata.read_h5ad(client_data_folder / "valid.h5ad")
    adata_local_valid = adata_local_valid[:, hvg_list].copy()

    # # ---------------------------------------------------------
    # # THIS BLOCK IS ONLY FOR TESTING AND WILL BE REMOVED LATER
    # # ---------------------------------------------------------

    # adata_local_train = load_local_data_simulation(
    #     client_id,
    #     "data_centralized/pancreas_train.h5ad",
    # )

    # adata_local_valid = load_local_data_simulation(
    #     client_id,
    #     "data_centralized/pancreas_valid.h5ad",
    # )

    # adata_local_train = adata_local_train[:, hvg_list].copy()
    # adata_local_valid = adata_local_valid[:, hvg_list].copy()

    # ---------------------------------------------------------
    # Build local SCVI model
    # ---------------------------------------------------------

    arch_cfg = get_architecture(context.run_config)

    scvi_model = create_scvi_model(
        adata_local_train,
        batch_list,
        arch_cfg,
    )

    return client_id, adata_local_train, adata_local_valid, scvi_model


# ---------------------------------------------------------------------
# Training endpoint
# ---------------------------------------------------------------------

@app.train()
def train(msg: Message, context: Context) -> Message:
    """
    Run one round of local client training.

    Flower server sends:
      - current global model weights
      - training config (number of local epochs)

    Client returns:
      - updated local model weights
      - number of local training examples
    """

    # Load local data + local model
    client_id, adata_local_train, _, scvi_model = _load_client_state(context)

    # Number of local epochs chosen by the server
    num_local_epochs = int(msg.content["config"]["num_local_epochs"])

    # Load global weights received from the server
    set_weights(
        scvi_model,
        msg.content["arrays"].to_numpy_ndarrays(),
    )

    # ---------------------------------------------------------
    # Local training step
    # ---------------------------------------------------------

    scvi_model.train(
        max_epochs=num_local_epochs,
        train_size=1.0,
    )

    # Extract updated model weights after local training
    updated_weights = get_weights(scvi_model)

    # Prepare reply to server
    content = RecordDict(
        {
            "arrays": ArrayRecord(
                numpy_ndarrays=updated_weights
            ),
            "metrics": MetricRecord(
                {
                    # Used for weighted averaging on the server (FedAvg)
                    "num-examples": int(adata_local_train.n_obs),
                }
            ),
        }
    )

    # ---------------------------------------------------------
    # Free memory after local training
    # ---------------------------------------------------------

    del scvi_model
    del adata_local_train
    del updated_weights

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return Message(content=content, reply_to=msg)


# ---------------------------------------------------------------------
# Evaluation endpoint
# ---------------------------------------------------------------------

@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """
    Evaluate current global model on local client data.

    The server sends global weights.
    The client computes:

      - train_loss
      - valid_loss

    and sends metrics back to the server.
    """

    # Load local data + model
    client_id, adata_local_train, adata_local_valid, scvi_model = _load_client_state(
        context
    )

    # Load current global model weights
    set_weights(
        scvi_model,
        msg.content["arrays"].to_numpy_ndarrays(),
    )

    print(
        f"Client ID: {client_id} | "
        f"Local valid data shape: {adata_local_valid.shape}"
    )

    # ---------------------------------------------------------
    # Compute local losses
    # ---------------------------------------------------------

    train_loss = get_loss(scvi_model, adata_local_train)
    valid_loss = get_loss(scvi_model, adata_local_valid)

    # Reply to server with metrics only
    content = RecordDict(
        {
            "metrics": MetricRecord(
                {
                    # Used for weighted averaging
                    "num-examples": int(adata_local_valid.n_obs),

                    # Local metrics
                    "train_loss": float(train_loss),
                    "valid_loss": float(valid_loss),

                    # Alias used by some strategies
                    "eval_loss": float(valid_loss),

                    # Useful for debugging
                    "client_id": int(client_id),
                }
            )
        }
    )

    # ---------------------------------------------------------
    # Free memory after evaluation
    # ---------------------------------------------------------

    del scvi_model
    del adata_local_train
    del adata_local_valid

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return Message(content=content, reply_to=msg)