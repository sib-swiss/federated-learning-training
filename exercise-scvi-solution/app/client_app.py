"""FedSCVI: A Flower for federated single cell variational inference."""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.client.mod import secaggplus_mod 

from app.task import load_local_data_simulation
from app.task import read_json, get_weights, set_weights, get_loss, get_architecture, create_scvi_model

import gc


# Define Flower Client
class ScviClient(NumPyClient):

    # Initilize Flower Client
    def __init__(
        self, 
        client_id, 
        adata_local_train,
        adata_local_valid, 
        scvi_model,
        num_local_epochs
    ):
        self.client_id = client_id
        self.adata_local_train = adata_local_train
        self.adata_local_valid = adata_local_valid
        self.scvi_model = scvi_model
        self.num_local_epochs = num_local_epochs

    def fit(self, parameters, config):
        set_weights(self.scvi_model, parameters)
        self.scvi_model.train(max_epochs=self.num_local_epochs)
        gc.collect()
        
        return get_weights(self.scvi_model), self.adata_local_train.n_obs, {}
    
    def evaluate(self, parameters, config):
        set_weights(self.scvi_model, parameters)
        train_loss = get_loss(self.scvi_model, self.adata_local_train)
        valid_loss = get_loss(self.scvi_model, self.adata_local_valid)
        loss_dict = {"train_loss": float(train_loss), "valid_loss": float(valid_loss)}

        gc.collect()

        return valid_loss, self.adata_local_valid.n_obs, loss_dict

def client_fn(context: Context):

    # Get the node id
    client_id = context.node_config["partition-id"]

    # Number of local epochs
    num_local_epochs = context.run_config.get("num_local_epochs")

    # Load HVG list and batch list
    hvg_list_path = context.run_config.get("hvg_list_path")
    batch_list_path = context.run_config.get("batch_list_path")
    hvg_list = read_json(hvg_list_path)
    batch_list = read_json(batch_list_path)

    # Retrieve the local adatas (train)
    adata_file_path = context.run_config["adata_train_file_path"]
    adata_local_train = load_local_data_simulation(client_id, adata_file_path)
    adata_local_train = adata_local_train[:, hvg_list].copy()

    # Retrieve the local adatas (valid)
    adata_file_path = context.run_config["adata_valid_file_path"]
    adata_local_valid = load_local_data_simulation(client_id, adata_file_path)
    adata_local_valid = adata_local_valid[:, hvg_list].copy()

    # Load model architecture parameters
    arch_cfg = get_architecture(context.run_config)

    # scVI model
    scvi_model = create_scvi_model(adata_local_train, batch_list, arch_cfg) 

    print(f"Client ID: {client_id} | Local train data shape: {adata_local_train.shape} | Local valid data shape: {adata_local_valid.shape}")
    
    return ScviClient(client_id, adata_local_train, adata_local_valid, scvi_model, num_local_epochs).to_client() 

#app = ClientApp(client_fn=client_fn) 
app = ClientApp(client_fn=client_fn, mods=[secaggplus_mod])  # To SecAgg+ on the client

