from __future__ import annotations

import json
from logging import INFO
from pathlib import Path
from typing import Iterable

import numpy as np

from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.common.logger import log
from flwr.serverapp.strategy import FedAvg

from app.task import compute_hvg_from_stats

class FedAvgComputeMaskSaveHvg(FedAvg):
    """
    Custom FedAvg strategy.

    Flower first averages the client statistics using FedAvg.
    Then we transform those aggregated statistics into a binary mask
    indicating the highly variable genes (HVGs).

    Unlike FedAvgWithResultSaving, this class does not save anything.
    It only computes and returns the mask.
    """

    def __init__(
        self,
        all_genes_file_path: str,
        hvg_file_path: str,
        *args,
        **kwargs,
    ) -> None:
        """
        Store the list of all genes once when the strategy is created.

        The HVG mask must be aligned with this list:
        mask[i] = 1 means all_genes_list[i] is selected.
        """
        super().__init__(*args, **kwargs)

        self.all_genes_file_path = Path(all_genes_file_path)
        self.hvg_file_path = Path(hvg_file_path)

        with open(self.all_genes_file_path, "r") as f:
            self.all_genes_list = json.load(f)

    def _compute_mask(
        self,
        arrays: ArrayRecord,
        num_clients: int,
    ) -> ArrayRecord:
        """
        Convert aggregated client statistics into an HVG mask.

        Clients send:
          1. gene-wise sums
          2. gene-wise squared sums
          3. number of local cells

        FedAvg averages these values, so here we reconstruct totals.
        """

        gene_sum_avg, gene_sum_sq_avg, n_cells_avg = arrays.to_numpy_ndarrays()
        n_cells_avg = float(np.asarray(n_cells_avg).flatten()[0])

        # FedAvg returns averages across clients.
        # Multiplying by the number of clients recovers totals.
        gene_sum = np.asarray(gene_sum_avg) * num_clients
        gene_sum_sq = np.asarray(gene_sum_sq_avg) * num_clients
        n_cells = n_cells_avg * num_clients

        # Compute HVGs from sufficient statistics.
        res = compute_hvg_from_stats(
            gene_sum,
            gene_sum_sq,
            n_cells,
            self.all_genes_list,
        )

        # Convert the boolean HVG column into a 0/1 mask.
        hvg_mask = res["highly_variable"].astype(np.int64).to_numpy()

        log(
            INFO,
            f"[FedAvgComputeMask] Computed HVG mask with "
            f"{int(hvg_mask.sum())} selected genes.",
        )

        # Extract the list of selected HVGs
        hvg_genes = res.index[res["highly_variable"]].tolist()

        # Save the list of HVGs for server-side use (e.g. downstream model definion)
        self.hvg_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.hvg_file_path, "w") as f:
            json.dump(hvg_genes, f, indent=2)

        log(
            INFO,
            f"Saved HVG list to {self.hvg_file_path}",
        )

        return ArrayRecord(numpy_ndarrays=[hvg_mask])

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """
        Called by Flower after clients complete one training round.

        Step 1:
            Run standard FedAvg aggregation on the client statistics.

        Step 2:
            Replace the aggregated statistics with the final HVG mask.
        """

        replies = list(replies)

        arrays, metrics = super().aggregate_train(server_round, replies)

        if arrays is None:
            return None, metrics

        mask_arrays = self._compute_mask(
            arrays=arrays,
            num_clients=len(replies),
        )

        return mask_arrays, metrics