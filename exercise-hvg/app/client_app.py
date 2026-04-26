"""HighVariance: Flower client app for highly variable gene identification."""

from pathlib import Path
import json
import anndata
import numpy as np

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from app.task import compute_hvg_client_stats

# Create Flower client application
app = ClientApp()


# ============================================================
# Trivial training endpoint
# ============================================================

@app.train()
def train(msg: Message, context: Context) -> Message:
    """
    Called by Flower during the training phase.

    This minimal example does not load any real data.
    Each client simply generates one random number and
    sends it to the server.

    The goal is only to understand how client-to-server
    communication works.
    """

    # --------------------------------------------------------
    # Generate a random local value
    # --------------------------------------------------------

    random_value = np.random.randint(0, 10)

    # --------------------------------------------------------
    # Prepare arrays to send to server
    # --------------------------------------------------------

    arrays = ArrayRecord(
        numpy_ndarrays=[
            np.array([random_value], dtype=np.float64)
        ]
    )

    # FedAvg requires a numeric weight.
    # Setting weight = 1 means all clients contribute equally.
    metrics = MetricRecord(
        {
            "client-weight": 1,
        }
    )

    # Build reply message
    content = RecordDict(
        {
            "arrays": arrays,
            "metrics": metrics,
        }
    )

    return Message(content=content, reply_to=msg)

# # ============================================================
# # Trivial evaluation endpoint
# # ============================================================

# @app.evaluate()
# def evaluate(msg: Message, context: Context) -> Message:
#     """
#     Called by Flower after server aggregation.

#     The server sends back the global result
#     (here: the average of all client values).

#     Each client receives the value, reads it,
#     and prints it locally.
#     """

#     # Unique client identifier
#     client_id = context.node_config["partition-id"]

#     # --------------------------------------------------------
#     # Read aggregated value sent by the server
#     # --------------------------------------------------------

#     # The server sends arrays in the same Flower format.
#     arrays = msg.content["arrays"]

#     # Extract first array, then first scalar value
#     received = arrays.to_numpy_ndarrays()[0]

#     # Convert NumPy scalar to standard Python float
#     average_value = float(received.flatten()[0])

#     # --------------------------------------------------------
#     # Print received global result
#     # --------------------------------------------------------

#     print(
#         f"Client ID: {client_id} | "
#         f"Received global average: {average_value:.4f}"
#     )

#     # --------------------------------------------------------
#     # Send optional acknowledgement metrics
#     # --------------------------------------------------------

#     metrics = MetricRecord(
#         {
#             "client-weight": 1,
#         }
#     )

#     content = RecordDict(
#         {
#             "metrics": metrics
#         }
#     )

#     return Message(content=content, reply_to=msg)

 
# ============================================================
# Actual training endpoint
# ============================================================

# @app.train()
# def train(msg: Message, context: Context) -> Message:
#     """
#     Called by Flower during the training phase.

#     In this preprocessing exercise, no neural network is trained.
#     Instead, each client computes local sufficient statistics
#     needed to identify highly variable genes globally.

#     The client sends:
#       - gene-wise sums
#       - gene-wise squared sums
#       - number of local cells
#     """

#     # Unique client identifier assigned by Flower
#     partition_id = context.node_config["partition-id"]

#     # Folder containing this client's local data
#     client_data_folder = Path(
#         context.run_config["client_folder_path"].format(
#             partition_id=partition_id
#         )
#     )

#     # --------------------------------------------------------
#     # Load gene list and data
#     # --------------------------------------------------------

#     # Ordered list of all genes.
#     # This order must be identical across all clients.
#     all_genes_file_path = client_data_folder / context.run_config["all_genes_file"]
#     all_genes_list = json.loads(all_genes_file_path.read_text())

#     # Load training data for the client
#     adata_local = anndata.read_h5ad(client_data_folder / "train.h5ad")

#     # --------------------------------------------------------
#     # Compute local hvg statistics
#     # --------------------------------------------------------

#     stats = compute_hvg_client_stats(
#         adata_local,
#         all_genes_list,
#     )

#     # --------------------------------------------------------
#     # Prepare arrays to send to server
#     # --------------------------------------------------------

#     arrays = ArrayRecord(
#         numpy_ndarrays=[
#             stats["sum"],                         # gene-wise sums
#             stats["sum_sq"],                     # gene-wise squared sums
#             np.array([stats["n_cells"]], dtype=np.float64),  # sample size
#         ]
#     )

#     # Metadata used by the server strategy
#     # This app used FedAvg but for summing statistics, the "client-weight" is set to 1 for all clients
#     metrics = MetricRecord(
#         {
#             "client-weight": 1,
#         }
#     )

#     content = RecordDict(
#         {
#             "arrays": arrays,
#             "metrics": metrics,
#         }
#     )

#     return Message(content=content, reply_to=msg)


# ============================================================
# Actual evaluation endpoint
# ============================================================

# @app.evaluate()
# def evaluate(msg: Message, context: Context) -> Message:
#     """
#     Called by Flower after server aggregation.

#     The server sends back the global HVG mask.
#     Each client converts the mask into a list of gene names
#     and saves it locally.
#     """

#     # Unique client identifier
#     client_id = context.node_config["partition-id"]

#     # Client folder
#     client_data_folder = Path(
#         context.run_config["client_folder_path"].format(
#             partition_id=client_id
#         )
#     ).expanduser().resolve()

#     # --------------------------------------------------------
#     # Load shared gene list
#     # --------------------------------------------------------

#     all_genes_file_path = (
#         client_data_folder /
#         Path(context.run_config["all_genes_file"])
#     )

#     all_genes_list = json.loads(all_genes_file_path.read_text())

#     # Output file name for selected genes
#     results_file_name = context.run_config.get("top2k_genes_file")
#     results_file_path = client_data_folder / results_file_name

#     # --------------------------------------------------------
#     # Read mask sent by server
#     # --------------------------------------------------------

#     arrays = msg.content["arrays"]

#     received = arrays.to_numpy_ndarrays()[0]
#     received = np.asarray(received).flatten()

#     # Safety check: mask length must match number of genes
#     if len(received) != len(all_genes_list):

#         print(
#             f"Client ID: {client_id} | "
#             f"Skipping evaluate because received array "
#             f"length is {len(received)} but expected "
#             f"{len(all_genes_list)}"
#         )

#         metrics = MetricRecord(
#             {
#                 "client-weight": 1,
#                 "saved": 0,
#                 "skipped": 1,
#                 "received-length": int(len(received)),
#             }
#         )

#         content = RecordDict({"metrics": metrics})
#         return Message(content=content, reply_to=msg)

#     # --------------------------------------------------------
#     # Convert mask -> gene names
#     # --------------------------------------------------------

#     hvg_mask = received.astype(int)

#     top_genes = [
#         gene
#         for gene, keep in zip(all_genes_list, hvg_mask)
#         if keep == 1
#     ]

#     # --------------------------------------------------------
#     # Save result locally
#     # --------------------------------------------------------

#     results_file_path.parent.mkdir(parents=True, exist_ok=True)

#     results_file_path.write_text(
#         json.dumps(top_genes, indent=2)
#     )

#     print(
#         f"Client ID: {client_id} | "
#         f"Saved {len(top_genes)} HVGs to "
#         f"{results_file_path}"
#     )

#     # Reply metrics
#     metrics = MetricRecord(
#         {
#             "client-weight": 1,
#             "saved": 1,
#             "skipped": 0,
#             "num-hvg": len(top_genes),
#         }
#     )

#     content = RecordDict({"metrics": metrics})

#     return Message(content=content, reply_to=msg)