"""HighVariance: A Flower for high-variance genes identification."""

from pathlib import Path
import json

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import anndata

from app.task import compute_hvg_client_stats

# Define Flower Client
class FlowerClient(NumPyClient):

    # Initilize Flower Client
    def __init__(self, 
        data,
        all_genes_list,
    ):
        self.data = data
        self.all_genes_list = all_genes_list

    # Fit function: compute the sufficient statistics for HVG calculation and return to server
    def fit(self, parameters, config):

        stats = compute_hvg_client_stats(self.data, self.all_genes_list)
        gene_sum = stats["sum"]
        gene_sum_sq = stats["sum_sq"]
        n_cells = stats["n_cells"]

        return [gene_sum, gene_sum_sq, n_cells], 1, {}      

def client_fn(context: Context):

    # Get the node id
    partition_id = context.node_config["partition-id"]

    # Get the client data folder
    client_data_folder = Path(context.run_config["client_folder_path"].format(partition_id=partition_id))

    # Load the shared genes list from config
    all_genes_file_path = Path(context.run_config["all_genes_file_path"])
    all_genes_list = json.loads(all_genes_file_path.read_text())

    # Load the local data for the client
    adata_local = anndata.read_h5ad(client_data_folder / "train.h5ad")
    print(f"Client ID: {partition_id} | Local data shape: {adata_local.shape}")
    
    return FlowerClient(adata_local, all_genes_list).to_client() 

# Flower ClientApp
app = ClientApp(client_fn=client_fn)