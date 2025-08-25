"""HighVariance: A Flower for high-variance genes identification."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
#from flwr.server.strategy import FedAvg
from flwr.common.logger import log

from app.task import get_dummy_start
from app.custom_strategy import FedAvgWithModelSaving

from logging import INFO
import json

def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Retrieve the configuration parameters from the context
    num_rounds = context.run_config["num-server-rounds"]
    genes_file_path = context.run_config["genes_file_path"]
    results_file_path=context.run_config["results_file_path"]

    # Dummy initial conditions (required by the strategy but not used in this task)
    initial_parameters = ndarrays_to_parameters([get_dummy_start()])

    # Read a JSON file with list of gene names
    with open(genes_file_path, "r") as f:
        genes_list = json.load(f)
    log(INFO, f"Loaded {len(genes_list)} genes.")

    # Define the strategy
    strategy = FedAvgWithModelSaving(
        # Use all clients
        fraction_fit=1.0,
        # Interrupt if any client fails
        accept_failures=False,
        # Disable evaluation
        fraction_evaluate=0.0,
        # Initial conditions
        initial_parameters=initial_parameters,
        # File to save the results
        results_file_path=results_file_path,
        # Genes list for filtering
        genes_list=genes_list
    )

    # Read from config
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)