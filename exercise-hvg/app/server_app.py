"""HighVariance: Flower client app for highly variable gene identification."""

from pathlib import Path
import numpy as np

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp

from app.custom_strategy import FedAvgComputeMaskSaveHvg

from flwr.serverapp.strategy import FedAvg

# Create Flower server application
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Main entry point of the ServerApp.

    The server coordinates the federated preprocessing workflow:
      1. Read configuration values
      2. Create dummy initial arrays
      3. Instantiate the custom aggregation strategy
      4. Start the federated run
    """

    # ============================================================
    # 1. Read configuration parameters
    # ============================================================

    # Server folder
    server_folder = Path(context.run_config["server_folder_path"])

    # File containing the ordered list of all genes
    all_genes_file_path = server_folder / context.run_config["all_genes_file"]

    # File containing the ordered list of all genes
    top2k_genes_file_path = server_folder / context.run_config["top2k_genes_file"]

    # ============================================================
    # 2. Create initial dummy arrays
    # ============================================================

    # Flower expects initial arrays even if no neural network is trained.
    # Here we use placeholder arrays with the correct dimensions.
    initial_arrays = ArrayRecord(
        numpy_ndarrays=[np.ones((1, 1))]
    )

    # ============================================================
    # 3. Aggregation strategy
    # ============================================================

    # strategy = FedAvg(
    #     fraction_train=1.0,
    #     fraction_evaluate=1.0,
    #     weighted_by_key="client-weight",
    # )

    # This is a custom strategy that extends FedAvg to compute an HVG mask.
    # It performs the following steps:
    #   - receives client sufficient statistics
    #   - uses FedAvg aggregation
    #   - reconstructs global statistics
    #   - computes the highly variable genes
    #   - returns a binary mask to clients
    strategy = FedAvgComputeMaskSaveHvg(
        all_genes_file_path=all_genes_file_path,
        hvg_file_path=top2k_genes_file_path,
        fraction_train=1.0,
        fraction_evaluate=1.0,
        weighted_by_key="client-weight",
    )

    # ============================================================
    # 4. Start federated preprocessing
    # ============================================================

    strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=1, # We only need 1 round for this preprocessing step
    )