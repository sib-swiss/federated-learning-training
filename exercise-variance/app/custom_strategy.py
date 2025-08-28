from flwr.server.strategy import FedAvg
from flwr.common import parameters_to_ndarrays
from flwr.common.logger import log

import numpy as np
import pandas as pd
from logging import INFO
from pathlib import Path
import json

class FedAvgWithModelSaving(FedAvg):
    """
    Custom FedAvg strategy that saves the global model state to disk at final server round.
    """
    def __init__(self, results_file_path: str, genes_list=None, *args, **kwargs):
        self.results_file_path = Path(results_file_path)
        self.genes_list = genes_list
        super().__init__(*args, **kwargs)

    def _save_results(self, parameters):
        """Save the results to disk as a JSON file."""
        json_filename = self.results_file_path

        ndarrays = parameters_to_ndarrays(parameters)
        flat_params = np.concatenate([arr.flatten() for arr in ndarrays])

        # Save as JSON list
        with open(json_filename, "w") as f:
            json.dump(flat_params.tolist(), f, indent=2)

        log(INFO, f"Results saved to: {json_filename}")

    # def _save_genes_list(self, parameters):
    #     """Save the list of top2000 genes to disk as a JSON file."""
    #     json_filename = self.results_file_path
    #     ndarrays = parameters_to_ndarrays(parameters)
    #     flat_params = np.concatenate([arr.flatten() for arr in ndarrays])

    #     # Use provided genes_list if valid, otherwise fallback to indices
    #     genes = (
    #         self.genes_list
    #         if self.genes_list and len(self.genes_list) == len(flat_params)
    #         else list(range(len(flat_params)))
    #     )

    #     # Sort by parameter values and take top2000
    #     top2000_genes = [
    #         gene for _, gene in sorted(zip(flat_params, genes), reverse=True)[:2000]
    #     ]

    #     # Write JSON
    #     with open(json_filename, "w") as f:
    #         json.dump(top2000_genes, f, indent=2)

    #     log(INFO, f"Results saved to: {json_filename}")

    def evaluate(self, server_round: int, parameters, *args, **kwargs):
        """Evaluate model parameters and save after each round."""
        if server_round != 0:
            self._save_results(parameters)
            #self._save_genes_list(parameters)
        return super().evaluate(server_round, parameters, *args, **kwargs)
