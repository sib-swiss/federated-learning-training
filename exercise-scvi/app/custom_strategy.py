"""Custom FedAvg strategies for FedSCVI using the Flower Message API.

The three classes below, all inheriting from FedAvg, are intentionally organized
as a progression:

1. FedAvgSaveModel
   Adds only one feature to FedAvg: save the final global model.

2. FedAvgSaveModelPlotLosses
   Tracks both train and validation losses and plots them.

3. FedAvgSaveModelPlotLossesEarlyStopping
   Extends Strategy 3 by adding early stopping based on validation loss.

All four strategies override the same two Flower hooks:

- aggregate_train(...)
  Called after clients finish local training. This is where we receive the
  aggregated model arrays and can save the global model.

- aggregate_evaluate(...)
  Called after clients finish evaluation. This is where we read client metrics
  and compute global weighted losses.
"""

from __future__ import annotations

import csv
import os

import numpy as np
from flwr.app import ArrayRecord
from flwr.serverapp.strategy import FedAvg


# ---------------------------------------------------------------------
# Strategy 1: Save the final global model
# ---------------------------------------------------------------------

class FedAvgSaveModel(FedAvg):
    """FedAvg that triggers a callback on the final round to persist the model."""

    def __init__(self, *, num_rounds: int, on_final_arrays, **kwargs):
        super().__init__(**kwargs)
        self._num_rounds = int(num_rounds)
        self._on_final_arrays = on_final_arrays

    def aggregate_train(self, server_round: int, replies):
        agg_arrays, metrics = super().aggregate_train(server_round, replies)

        if agg_arrays is not None and int(server_round) >= self._num_rounds:
            try:
                self._on_final_arrays(agg_arrays)
            except Exception as e:
                print(f"[WARN] Failed saving on final round: {e}")

        return agg_arrays, metrics


# ---------------------------------------------------------------------
# Strategy 2: Save the final model + track train/validation loss
# ---------------------------------------------------------------------

class FedAvgSaveModelPlotLosses(FedAvg):
    """
    FedAvg that:
      - saves final aggregated arrays at the last round
      - logs weighted train/validation losses per round
      - regenerates a PNG plot every round
    """

    def __init__(
        self,
        *,
        num_rounds: int,
        on_final_arrays,
        loss_history_path: str | None = "./model_federated/global_loss.csv",
        loss_plot_path: str = "./model_federated/global_loss.png",
        loss_numpy_path: str | None = "./model_federated/global_loss.npy",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._num_rounds = int(num_rounds)
        self._on_final_arrays = on_final_arrays
        self.loss_history_path = loss_history_path
        self.loss_plot_path = loss_plot_path
        self.loss_numpy_path = loss_numpy_path

        os.makedirs(os.path.dirname(self.loss_plot_path) or ".", exist_ok=True)

        if self.loss_history_path is not None:
            os.makedirs(os.path.dirname(self.loss_history_path) or ".", exist_ok=True)
            with open(self.loss_history_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "round",
                        "global_weighted_train_loss",
                        "global_weighted_valid_loss",
                    ],
                )
                writer.writeheader()

        if self.loss_numpy_path is not None:
            os.makedirs(os.path.dirname(self.loss_numpy_path) or ".", exist_ok=True)

        self._mpl_ready = False
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: F401

            self._mpl_ready = True
        except Exception as e:
            print(f"[WARN] Matplotlib not available or failed to init: {e}")

        self._history = {"rounds": [], "train": [], "valid": []}

    def _append_loss(self, rnd: int, train_value: float, valid_value: float) -> None:
        self._history["rounds"].append(int(rnd))
        self._history["train"].append(float(train_value))
        self._history["valid"].append(float(valid_value))

        if self.loss_history_path is None:
            return

        try:
            with open(self.loss_history_path, "a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "round",
                        "global_weighted_train_loss",
                        "global_weighted_valid_loss",
                    ],
                )
                writer.writerow(
                    {
                        "round": int(rnd),
                        "global_weighted_train_loss": float(train_value),
                        "global_weighted_valid_loss": float(valid_value),
                    }
                )
        except Exception as e:
            print(f"[WARN] Could not write loss history: {e}")

    def _update_plot(self) -> None:
        if not self._mpl_ready:
            return

        try:
            import matplotlib.pyplot as plt

            rounds = list(self._history["rounds"])
            train = list(self._history["train"])
            valid = list(self._history["valid"])

            if not rounds:
                return

            plt.figure()
            plt.plot(rounds, train, marker="o", label="Train loss")
            plt.plot(rounds, valid, marker="o", label="Validation loss")
            plt.xlabel("Round")
            plt.ylabel("Global weighted loss")
            plt.title("Federated train vs validation loss")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.loss_plot_path, dpi=150)
            plt.close()
            print(f"[INFO] Updated loss plot at {self.loss_plot_path}")

            if self.loss_numpy_path is not None:
                loss_array = np.array([rounds, train, valid])
                np.save(self.loss_numpy_path, loss_array)
                print(f"[INFO] Saved loss array to {self.loss_numpy_path}")

        except Exception as e:
            print(f"[WARN] Could not update loss plot: {e}")

    def aggregate_train(self, server_round: int, replies):
        aggregated_arrays, aggregated_metrics = super().aggregate_train(
            server_round, replies
        )

        if aggregated_arrays is not None and int(server_round) >= self._num_rounds:
            try:
                self._on_final_arrays(aggregated_arrays)
            except Exception as e:
                print(f"[WARN] Failed saving on final round: {e}")

        return aggregated_arrays, aggregated_metrics

    def aggregate_evaluate(self, server_round: int, replies):
        aggregated_metrics = super().aggregate_evaluate(server_round, replies)

        total_examples = 0
        train_weighted_sum = 0.0
        valid_weighted_sum = 0.0
        have_train = False
        have_valid = False

        for reply_msg in replies:
            if not reply_msg.has_content():
                continue

            msg_metrics = reply_msg.content.get("metrics")
            if msg_metrics is None:
                continue

            try:
                n_i = int(msg_metrics["num-examples"])
            except (KeyError, TypeError):
                n_i = 0

            if n_i <= 0:
                continue

            total_examples += n_i

            try:
                train_weighted_sum += float(msg_metrics["train_loss"]) * n_i
                have_train = True
            except (KeyError, TypeError):
                pass

            try:
                valid_weighted_sum += float(msg_metrics["eval_loss"]) * n_i
                have_valid = True
            except (KeyError, TypeError):
                try:
                    valid_weighted_sum += float(msg_metrics["valid_loss"]) * n_i
                    have_valid = True
                except (KeyError, TypeError):
                    pass

        weighted_train = (
            train_weighted_sum / total_examples
            if have_train and total_examples > 0
            else None
        )
        weighted_valid = (
            valid_weighted_sum / total_examples
            if have_valid and total_examples > 0
            else None
        )

        if weighted_train is not None or weighted_valid is not None:
            tr = float(weighted_train) if weighted_train is not None else float("nan")
            va = float(weighted_valid) if weighted_valid is not None else float("nan")

            print(f"[INFO] Round {server_round}: train={tr:.6f} | valid={va:.6f}")

            self._append_loss(server_round, tr, va)
            self._update_plot()

        return aggregated_metrics
    

# ---------------------------------------------------------------------
# Strategy 2: Strategy 2 + early stopping
# ---------------------------------------------------------------------

class EarlyStopException(Exception):
    """Raised internally to stop Flower training early."""
    pass

class FedAvgSaveModelPlotLossesEarlyStopping(FedAvgSaveModelPlotLosses):
    def __init__(
        self,
        *,
        early_stopping_patience: int = 3,
        early_stopping_min_delta: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._es_patience = int(early_stopping_patience)
        self._es_min_delta = float(early_stopping_min_delta)
        self._es_best_valid_loss: float | None = None
        self._es_wait = 0
        self._es_best_arrays: ArrayRecord | None = None
        self._es_latest_arrays: ArrayRecord | None = None
        self._es_stopped = False
        self._es_stop_round: int | None = None

    def _check_early_stopping(
        self,
        server_round: int,
        valid_loss: float,
        latest_arrays: ArrayRecord | None,
    ) -> None:
        if self._es_patience <= 0 or self._es_stopped:
            return

        improved = (
            self._es_best_valid_loss is None
            or valid_loss < self._es_best_valid_loss - self._es_min_delta
        )

        if improved:
            self._es_best_valid_loss = valid_loss
            self._es_wait = 0

            if latest_arrays is not None:
                self._es_best_arrays = latest_arrays

            print(
                f"[EARLY STOP] Round {server_round}: "
                f"new best valid loss = {valid_loss:.6f}"
            )
            return

        self._es_wait += 1

        print(
            f"[EARLY STOP] Round {server_round}: no improvement for "
            f"{self._es_wait}/{self._es_patience} rounds "
            f"(best={self._es_best_valid_loss:.6f}, current={valid_loss:.6f})"
        )

        if self._es_wait >= self._es_patience:
            self._es_stopped = True
            self._es_stop_round = server_round

            print(f"[EARLY STOP] Triggered at round {server_round}")

            if self._es_best_arrays is not None:
                try:
                    self._on_final_arrays(self._es_best_arrays)
                    print("[EARLY STOP] Saved best model.")
                except Exception as e:
                    print(f"[WARN] Failed saving best model on early stop: {e}")

            raise EarlyStopException()

    def aggregate_train(self, server_round: int, replies):
        aggregated_arrays, aggregated_metrics = FedAvgSaveModelPlotLosses.aggregate_train(
            self,
            server_round,
            replies,
        )

        if aggregated_arrays is not None:
            self._es_latest_arrays = aggregated_arrays

        return aggregated_arrays, aggregated_metrics

    def aggregate_evaluate(self, server_round: int, replies):
        aggregated_metrics = FedAvgSaveModelPlotLosses.aggregate_evaluate(
            self,
            server_round,
            replies,
        )

        valid_losses = self._history["valid"]

        if valid_losses:
            latest_valid_loss = float(valid_losses[-1])

            if not np.isnan(latest_valid_loss):
                self._check_early_stopping(
                    server_round=server_round,
                    valid_loss=latest_valid_loss,
                    latest_arrays=self._es_latest_arrays,
                )

        return aggregated_metrics