"""HighVariance: A Flower for high-variance genes identification."""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import numpy as np

from app.task import get_random_integer, load_local_data

# Define Flower Client
class FlowerClient(NumPyClient):

    # Initilize Flower Client
    def __init__(self, 
        max_rand_int, 
        max_rand_weight,
    ):
        self.max_rand_int = max_rand_int
        self.max_rand_weight = max_rand_weight

    # Generate a random integer and a random weight locally
    def fit(self, parameters, config):
        random_int = get_random_integer(self.max_rand_int)
        random_weight = get_random_integer(self.max_rand_weight)
        
        return [random_int], random_weight, {}      

def client_fn(context: Context):

    # Retrieve the max_rand_int and max_rand_weight from the context
    max_rand_int = context.run_config["max_rand_int"]
    max_rand_weight = context.run_config["max_rand_weight"]

    # Retrieve the adata file path from the context
    #adata_file_path = context.run_config["adata_file_path"]

    # Get the node id
    #partition_id = context.node_config["partition-id"]

    # Load the local data for the client
    #adata_local = load_local_data(partition_id, adata_file_path)
    #print(f"Client ID: {partition_id} | Local data shape: {adata_local.shape}")
    
    return FlowerClient(max_rand_int, max_rand_weight).to_client() 

# Flower ClientApp
app = ClientApp(client_fn=client_fn)