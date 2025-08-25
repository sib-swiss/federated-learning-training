"""FedSCVI: A Flower for federated single cell variational inference."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from flwr.server.strategy import FedAvg
from app.task import read_json, get_weights, set_weights, create_dummy_adata_from_hvg, get_architecture, create_scvi_model 


def server_fn(context: Context):
    """Build and configure the ServerApp components."""

    # --- Read runtime configuration from the Flower context ---
    num_rounds = int(context.run_config.get("num_rounds"))
    hvg_list_path = context.run_config.get("hvg_list_path")
    batch_list_path = context.run_config.get("batch_list_path")
    model_file_path = context.run_config.get("model_file_path")
    loss_plot_path = context.run_config.get("loss_plot_path")

    # --- Load gene/batch metadata ---
    hvg_list = read_json(hvg_list_path)
    batch_list = read_json(batch_list_path)

    # --- Initialize model and convert initial weights to Flower Parameters ---
    adata_dummy = create_dummy_adata_from_hvg(hvg_list, batch_list) # The easiest way is through a dummy AnnData
    arch_cfg = get_architecture(context.run_config)  # Validate architecture parameters
    scvi_model = create_scvi_model(adata_dummy, batch_list, arch_cfg)  # Validate model creation
    initial_parameters = ndarrays_to_parameters(get_weights(scvi_model))

    # FedAvg as a starting srategy
    strategy = FedAvg(
        fraction_fit=1.0,
        accept_failures=False,
        fraction_evaluate=1.0,
        initial_parameters=initial_parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

# Instantiate the ServerApp
app = ServerApp(server_fn=server_fn)


# from flwr.server import ServerApp, ServerAppComponents, ServerConfig, LegacyContext
# from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

# # Run SecAgg+ workflow from main
# @app.main()
# def main(grid, context: Context) -> None:
#     # Reuse the same strategy/config as in server_fn
#     comps = server_fn(context)  # gives us comps.strategy and comps.config

#     # LegacyContext is what DefaultWorkflow expects
#     legacy = LegacyContext(
#         context=context,
#         config=comps.config,
#         strategy=comps.strategy,
#         # client_manager can be omitted; it’s taken from `context`
#     )

#     # Read SecAgg+ knobs from run_config (add to pyproject.toml if you want)
#     num_shares = int(context.run_config.get("num_shares"))
#     reconstruction_threshold = int(context.run_config.get("reconstruction_threshold"))

#     workflow = DefaultWorkflow(
#         fit_workflow=SecAggPlusWorkflow(
#             num_shares=num_shares,
#             reconstruction_threshold=reconstruction_threshold,
#         )
#     )

#     # Run the workflow (this executes training rounds with SecAgg+)
#     workflow(grid, legacy)