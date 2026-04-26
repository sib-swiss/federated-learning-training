"""HighVariance: A Flower for high-variance genes identification."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
#from flwr.server.strategy import FedAvg

from app.task import get_dummy_start
from app.custom_strategy import FedAvgWithModelSaving


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Retrieve the configuration parameters from the context
    num_rounds = context.run_config["num-server-rounds"]
    all_genes_file_path = context.run_config["all_genes_file_path"]
    results_file_path=context.run_config["results_file_path"]

    # Dummy initial conditions (required by the strategy but not used in this task)
    initial_parameters = ndarrays_to_parameters([get_dummy_start()])

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
        # File with list of all genes (for ordering)
        all_genes_file_path=all_genes_file_path,
        # File to save the results
        results_file_path=results_file_path,
    )

    # Read from config
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)