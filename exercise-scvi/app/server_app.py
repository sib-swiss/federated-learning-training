"""FedSCVI: A Flower for federated single cell variational inference."""

from pathlib import Path

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from app.custom_strategy import (
    FedAvgSaveModel,
    FedAvgSaveModelPlotLosses,
    FedAvgSaveModelPlotLossesEarlyStopping,
    EarlyStopException,
)
from app.task import (
    create_dummy_adata_from_hvg,
    create_scvi_model,
    get_architecture,
    get_weights,
    make_on_final_arrays,
    read_json,
    set_weights,
)

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # ============================================================
    # 1. Read configuration parameters
    # ============================================================

    # Training hyperparameters
    num_rounds = int(context.run_config["num_rounds"])
    num_local_epochs = int(context.run_config["num_local_epochs"])
    early_stopping_patience = int(context.run_config.get("early_stopping_patience")) # Only used by Strategy 3
    early_stopping_min_delta = float(context.run_config.get("early_stopping_min_delta")) # Only used by Strategy 3

    # File paths
    server_folder = Path(context.run_config["server_folder_path"])
    top2k_genes_file_path = server_folder / context.run_config["top2k_genes_file"]
    all_techs_file_path = server_folder / context.run_config["all_techs_file"]
    model_file_path = context.run_config["model_file_path"]
    loss_plot_path = context.run_config.get("loss_plot_path", "loss_plot.png") # Only used by Strategies 2 and 3

    # ============================================================
    # 2. Build the initial global SCVI model
    # ============================================================
    
    # Load HVG list and techs list
    hvg_list = read_json(top2k_genes_file_path)
    batch_list = read_json(all_techs_file_path)
    
    # Note that initialization is performed through a dummy AnnData with the correct genes and batches
    adata_dummy = create_dummy_adata_from_hvg(hvg_list, batch_list)
    arch_cfg = get_architecture(context.run_config)
    scvi_model = create_scvi_model(adata_dummy, batch_list, arch_cfg)

    # Get initial global model weights as NumPy arrays to send to clients
    initial_arrays = ArrayRecord(numpy_ndarrays=get_weights(scvi_model))


    # ============================================================
    # 3. Strategy selection: uncomment exactly one strategy to run
    # ============================================================

    # Utility function for saving model (for strategies 1-3)
    on_final = make_on_final_arrays(
        model=scvi_model,
        set_weights=set_weights,
        save_path=model_file_path,
    )

    # Strategy 0: plain FedAvg
    #   - trains federated model
    #   - does not save final model
    #   - does not evaluate clients
    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        weighted_by_key="num-examples",
    )

    # # Strategy 1: FedAvg + save final model
    # #   - saves final aggregated SCVI model
    # strategy = FedAvgSaveModel(
    #     num_rounds=num_rounds,
    #     on_final_arrays=on_final,
    #     fraction_train=1.0,
    #     fraction_evaluate=0.0,
    #     weighted_by_key="num-examples",
    # )

    # # Strategy 2: FedAvg + train/valid loss plot 
    # #   - tracks train and validation losses
    # strategy = FedAvgSaveModelPlotLosses(
    #     num_rounds=num_rounds,
    #     on_final_arrays=on_final,
    #     fraction_train=1.0,
    #     fraction_evaluate=1.0,
    #     weighted_by_key="num-examples",
    #     loss_history_path=None,
    #     loss_plot_path=loss_plot_path,
    # )

    # # Strategy 3: FedAvg + train/valid loss plot + early stopping
    # #   - tracks train and validation losses
    # #   - optionally stops early
    # strategy = FedAvgSaveModelPlotLossesEarlyStopping(
    #     num_rounds=num_rounds,
    #     on_final_arrays=on_final,
    #     fraction_train=1.0,
    #     fraction_evaluate=1.0,
    #     weighted_by_key="num-examples",
    #     loss_history_path=None,
    #     loss_plot_path=loss_plot_path,
    #     early_stopping_patience=early_stopping_patience,
    #     early_stopping_min_delta=early_stopping_min_delta,
    # )


    # ============================================================
    # 4. Start federated training
    # ============================================================

    try:
        strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            train_config=ConfigRecord({"num_local_epochs": num_local_epochs}),
            num_rounds=num_rounds,
            evaluate_fn=None,
        )
    # This is raised by Strategy 3 when early stopping criteria are met
    except EarlyStopException:
        print("[EARLY STOP] Flower run stopped early.")