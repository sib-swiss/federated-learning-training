from __future__ import annotations

import json
from logging import INFO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from app.task import compute_hvg_from_stats


class FedAvgWithModelSaving(FedAvg):
    """Custom FedAvg strategy that saves aggregated gene results to disk."""

    def __init__(
        self,
        all_genes_file_path: str,
        results_file_path: str,
        *args,
        **kwargs,
    ) -> None:
        self.all_genes_file_path = Path(all_genes_file_path)
        self.results_file_path = Path(results_file_path)
        self.num_clients = 0
        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results and store the number of participating clients."""
        self.num_clients = len(results)
        return super().aggregate_fit(server_round, results, failures)

    def _save_results(self, parameters: Parameters) -> None:
        """Save the list of top highly variable genes to disk as JSON."""
        ndarrays = parameters_to_ndarrays(parameters)

        # FedAvg returns averaged parameters
        avg_gene_sum, avg_gene_sum_sq, avg_n_cells = ndarrays

        # Recover global sums from averages
        gene_sum = avg_gene_sum * self.num_clients
        gene_sum_sq = avg_gene_sum_sq * self.num_clients
        n_cells = avg_n_cells * self.num_clients

        with open(self.all_genes_file_path, "r") as f:
            all_genes_list = json.load(f)

        res = compute_hvg_from_stats(
            gene_sum,
            gene_sum_sq,
            n_cells,
            all_genes_list,
        )

        log(INFO, "Running smooth")

        top2k_genes = res[res["highly_variable"]].index.tolist()

        self.results_file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.results_file_path, "w") as f:
            json.dump(top2k_genes, f, indent=2)

        log(INFO, f"Results saved to: {self.results_file_path}")

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        *args,
        **kwargs,
    ):
        """Evaluate model parameters and save results after each round."""
        if server_round != 0:
            self._save_results(parameters)

        return super().evaluate(server_round, parameters, *args, **kwargs)