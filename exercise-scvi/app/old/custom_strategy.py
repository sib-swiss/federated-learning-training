from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import FedAvg

import os
import torch
import csv


##### Custom strategy that saves the final model 

    # # Create a callback to persist the final aggregated parameters
    # on_final = make_on_final_parameters(
    #     model=scvi_model,
    #     set_weights=set_weights,
    #     save_path=model_file_path,
    # )

    # strategy = SaveOnFinalFedAvg(
    #     num_rounds=num_rounds,
    #     on_final_parameters=on_final,
    #     fraction_fit=1.0,
    #     accept_failures=False,
    #     fraction_evaluate=0.0,
    #     initial_parameters=initial_parameters,
    # )

def save_final_model(model, path: str = "./model.pt") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    to_save = model.module if hasattr(model, "module") else model
    torch.save(to_save.state_dict(), path)
    print(f"[INFO] Saved final model to {path}")


def make_on_final_parameters(model, set_weights, save_path: str = "./model.pt"):
    """Return a callback that applies aggregated params to `model` and saves it."""
    def _on_final_parameters(final_parameters):
        final_weights = parameters_to_ndarrays(final_parameters)
        set_weights(model, final_weights)
        if hasattr(model, "is_trained"):
            try:
                model.is_trained = True
            except Exception:
                pass
        save_final_model(model, save_path)
    return _on_final_parameters


class SaveOnFinalFedAvg(FedAvg):
    """FedAvg that triggers a callback on the final round to persist the model."""
    def __init__(self, *, num_rounds: int, on_final_parameters, **kwargs):
        super().__init__(**kwargs)
        self._num_rounds = int(num_rounds)
        self._on_final_parameters = on_final_parameters

    def aggregate_fit(self, server_round, results, failures):
        agg_params, metrics = super().aggregate_fit(server_round, results, failures)
        if agg_params is not None and int(server_round) >= self._num_rounds:
            try:
                self._on_final_parameters(agg_params)
            except Exception as e:
                print(f"[WARN] Failed saving on final round: {e}")
        return agg_params, metrics


##### Custom strategy that saves the final model and plots the global loss, updating it every round

    # strategy = StoreLossSaveOnFinalFedAvg(
    #     num_rounds=num_rounds,
    #     on_final_parameters=on_final,
    #     fraction_fit=1.0,
    #     fraction_evaluate=1.0,         
    #     accept_failures=False,
    #     loss_plot_path=loss_plot_path,
    #     initial_parameters=initial_parameters,
    # )

class StoreLossSaveOnFinalFedAvg(FedAvg):
    """
    FedAvg that:
      - Saves the final aggregated parameters at the last round
      - Logs the global (weighted) validation loss per round to CSV
      - Regenerates a PNG plot of the global loss at every round
    """

    def __init__(
        self,
        *,
        num_rounds: int,
        on_final_parameters,
        loss_history_path: str = "./global_loss.csv",
        loss_plot_path: str = "./global_loss.png",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._num_rounds = int(num_rounds)
        self._on_final_parameters = on_final_parameters
        self.loss_history_path = loss_history_path
        self.loss_plot_path = loss_plot_path

        # Ensure parent dirs exist
        os.makedirs(os.path.dirname(self.loss_history_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.loss_plot_path) or ".", exist_ok=True)

        # (re)create CSV with header
        with open(self.loss_history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "global_weighted_loss"])
            writer.writeheader()

        # Try to set a non-interactive backend for headless environments
        self._mpl_ready = False
        try:
            import matplotlib
            matplotlib.use("Agg")  # safe in headless envs
            import matplotlib.pyplot as plt  # noqa: F401
            self._mpl_ready = True
        except Exception as e:
            print(f"[WARN] Matplotlib not available or failed to init: {e}")

    def _append_loss(self, rnd: int, loss_value: float) -> None:
        try:
            with open(self.loss_history_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["round", "global_weighted_loss"])
                writer.writerow({"round": int(rnd), "global_weighted_loss": float(loss_value)})
        except Exception as e:
            print(f"[WARN] Could not write loss history: {e}")

    def _update_plot(self) -> None:
        if not self._mpl_ready:
            return
        try:
            rounds, losses = [], []
            with open(self.loss_history_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rounds.append(int(row["round"]))
                    losses.append(float(row["global_weighted_loss"]))

            if not rounds:
                return

            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(rounds, losses, marker="o")
            plt.xlabel("Round")
            plt.ylabel("Global weighted validation loss")
            plt.title("Federated loss over rounds")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.loss_plot_path, dpi=150)
            plt.close()
            print(f"[INFO] Updated loss plot at {self.loss_plot_path}")
        except Exception as e:
            print(f"[WARN] Could not update loss plot: {e}")

    def aggregate_fit(self, server_round, results, failures):
        agg_params, metrics = super().aggregate_fit(server_round, results, failures)
        if agg_params is not None and int(server_round) >= self._num_rounds:
            try:
                self._on_final_parameters(agg_params)
            except Exception as e:
                print(f"[WARN] Failed saving on final round: {e}")
        return agg_params, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Compute the weighted average validation loss across clients, log it, and update the plot.
        """
        aggregated_loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Defensive recomputation if needed
        weighted_loss = None
        try:
            if results:
                total_examples = 0
                weighted_sum = 0.0
                for _, eval_res in results:
                    loss_i = float(eval_res.loss)
                    n_i = int(getattr(eval_res, "num_examples", 0))
                    if n_i > 0:
                        total_examples += n_i
                        weighted_sum += loss_i * n_i
                if total_examples > 0:
                    weighted_loss = weighted_sum / total_examples
        except Exception as e:
            print(f"[WARN] Could not compute weighted loss: {e}")

        if aggregated_loss is not None:
            weighted_loss = float(aggregated_loss)

        if weighted_loss is not None:
            print(f"[INFO] Round {server_round}: global weighted val loss = {weighted_loss:.6f}")
            self._append_loss(server_round, weighted_loss)
            self._update_plot()  # <— regenerate PNG each round

        return aggregated_loss, metrics



##### Custom strategy that saves the final model and plots both global train and validation losses, updating them every round

    # strategy = StoreBothLossesSaveOnFinalFedAvg(
    #     num_rounds=num_rounds,
    #     on_final_parameters=on_final,
    #     fraction_fit=1.0,
    #     accept_failures=False,
    #     fraction_evaluate=1.0,
    #     loss_history_path=None, # use in case you want to also save a CSV             
    #     loss_plot_path=loss_plot_path,
    #     initial_parameters=initial_parameters,
    # )

class StoreBothLossesSaveOnFinalFedAvg(FedAvg):
    """
    FedAvg that:
      - Saves the final aggregated parameters at the last round
      - Logs weighted TRAIN and VALID losses per round
      - Regenerates a PNG plot with both curves every round

    Notes:
      * If `loss_history_path=None`, no CSV is written (plot only).
    """

    def __init__(
        self,
        *,
        num_rounds: int,
        on_final_parameters,
        loss_history_path: str | None = "./global_loss.csv",
        loss_plot_path: str = "./global_loss.png",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._num_rounds = int(num_rounds)
        self._on_final_parameters = on_final_parameters
        self.loss_history_path = loss_history_path
        self.loss_plot_path = loss_plot_path

        # Ensure parent dirs exist
        os.makedirs(os.path.dirname(self.loss_plot_path) or ".", exist_ok=True)
        if self.loss_history_path is not None:
            os.makedirs(os.path.dirname(self.loss_history_path) or ".", exist_ok=True)
            # (re)create CSV with header
            with open(self.loss_history_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["round", "global_weighted_train_loss", "global_weighted_valid_loss"],
                )
                writer.writeheader()

        # Matplotlib for headless envs
        self._mpl_ready = False
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: F401
            self._mpl_ready = True
        except Exception as e:
            print(f"[WARN] Matplotlib not available or failed to init: {e}")

        # In-memory history (so CSV is optional)
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
                    fieldnames=["round", "global_weighted_train_loss", "global_weighted_valid_loss"],
                )
                writer.writerow({
                    "round": int(rnd),
                    "global_weighted_train_loss": float(train_value),
                    "global_weighted_valid_loss": float(valid_value),
                })
        except Exception as e:
            print(f"[WARN] Could not write loss history: {e}")

    def _update_plot(self) -> None:
        if not self._mpl_ready:
            return
        try:
            import matplotlib.pyplot as plt

            rounds = self._history["rounds"]
            train = self._history["train"]
            valid = self._history["valid"]
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
        except Exception as e:
            print(f"[WARN] Could not update loss plot: {e}")

    def aggregate_fit(self, server_round, results, failures):
        agg_params, metrics = super().aggregate_fit(server_round, results, failures)
        if agg_params is not None and int(server_round) >= self._num_rounds:
            try:
                self._on_final_parameters(agg_params)
            except Exception as e:
                print(f"[WARN] Failed saving on final round: {e}")
        return agg_params, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Compute weighted average train & validation losses across clients, log, and plot.
        """
        aggregated_valid_loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        total_examples = 0
        train_weighted_sum = 0.0
        valid_weighted_sum = 0.0
        have_train = False
        have_valid = False

        try:
            for _, eval_res in results:
                n_i = int(getattr(eval_res, "num_examples", 0))
                if n_i <= 0:
                    continue
                total_examples += n_i

                # Prefer metrics if present; fall back to eval_res.loss for valid
                m = getattr(eval_res, "metrics", {}) or {}
                if "train_loss" in m:
                    train_weighted_sum += float(m["train_loss"]) * n_i
                    have_train = True

                if "valid_loss" in m:
                    valid_weighted_sum += float(m["valid_loss"]) * n_i
                    have_valid = True
                else:
                    # Fallback: Flower evaluate loss
                    if eval_res.loss is not None:
                        valid_weighted_sum += float(eval_res.loss) * n_i
                        have_valid = True
        except Exception as e:
            print(f"[WARN] Could not compute weighted losses: {e}")

        weighted_train = (train_weighted_sum / total_examples) if (have_train and total_examples > 0) else None

        # If FedAvg already produced an aggregated validation loss, prefer it
        if aggregated_valid_loss is not None:
            weighted_valid = float(aggregated_valid_loss)
        else:
            weighted_valid = (valid_weighted_sum / total_examples) if (have_valid and total_examples > 0) else None

        if weighted_train is not None or weighted_valid is not None:
            tr = float(weighted_train) if weighted_train is not None else float("nan")
            va = float(weighted_valid) if weighted_valid is not None else float("nan")
            print(f"[INFO] Round {server_round}: train={tr:.6f} | valid={va:.6f}")
            # Use NaN-safe defaults
            self._append_loss(server_round, tr, va)
            self._update_plot()

        return aggregated_valid_loss, metrics