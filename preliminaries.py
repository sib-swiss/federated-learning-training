import json
import os
import subprocess
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split

import importlib.util
import subprocess
import sys
import shutil

# Small environment fixes
required = {
    "jax": "jax[cpu]==0.4.35",
    "numpyro": "numpyro==0.15.3",
    "gdown": "gdown==4.7.1"
}

missing = [
    pkg_spec
    for module_name, pkg_spec in required.items()
    if importlib.util.find_spec(module_name) is None
]

if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

import subprocess
import sys
import gdown

folders = [
    "data_centralized",
    "data_federated",
    "model_centralized",
    "model_federated",
]

files = [
    "loss.png",
    "global_loss.csv",
    "global_loss.npy",
]

# # Delete folders if they exist
# for folder in folders:
#     path = Path(folder)
#     if path.exists() and path.is_dir():
#         shutil.rmtree(path)
#         print(f"Deleted folder: {path}")

# # Delete files if they exist
# for file in files:
#     path = Path(file)
#     if path.exists() and path.is_file():
#         path.unlink()
#         print(f"Deleted file: {path}")


data_dir = Path("data_centralized")
data_dir.mkdir(exist_ok=True)

pancreas_adata_path = data_dir / "pancreas_full.h5ad"
# file_id = "1r_P3O8DE6Qu-6U0sFyZh1P_H6-un9mHc"

# gdown.download(
#     f"https://drive.google.com/uc?id={file_id}",
#     str(pancreas_adata_path),
#     quiet=False,
# )

pancreas_adata = sc.read_h5ad(pancreas_adata_path)

train_path = data_dir / "pancreas_train.h5ad"
valid_path = data_dir / "pancreas_valid.h5ad"
test_path = data_dir / "pancreas_test.h5ad"

# Split dataset by technology: keep smartseq2/celseq2 as held-out test
query_mask = pancreas_adata.obs["tech"].isin(["smartseq2", "celseq2"]).to_numpy()
pancreas_no_test = pancreas_adata[~query_mask].copy()
pancreas_test = pancreas_adata[query_mask].copy()

# 80/20 train/valid split on the remaining data, stratified by technology
y = pancreas_no_test.obs["tech"].astype("category")
indices = np.arange(pancreas_no_test.n_obs)

idx_train, idx_valid = train_test_split(
    indices,
    test_size=0.20,
    train_size=0.80,
    random_state=42,
    shuffle=True,
    stratify=y,
)

pancreas_train = pancreas_no_test[idx_train].copy()
pancreas_valid = pancreas_no_test[idx_valid].copy()

# Save centralized splits
pancreas_train.write(train_path)
pancreas_valid.write(valid_path)
pancreas_test.write(test_path)

print(
    f"Train: {pancreas_train.n_obs} cells | "
    f"Valid: {pancreas_valid.n_obs} cells | "
    f"Test: {pancreas_test.n_obs} cells"
)

# Print counts per technology
print("\nCells per technology:")
for name, adata in [
    ("Train", pancreas_train),
    ("Valid", pancreas_valid),
    ("Test", pancreas_test),
]:
    counts = adata.obs["tech"].value_counts().sort_index()
    print(f"\n{name} split:")
    for tech, n in counts.items():
        print(f"  {tech}: {n}")

all_genes = pancreas_adata.var_names.tolist()
techs = pancreas_adata.obs["tech"].unique().tolist()

all_genes_path = data_dir / "all_genes.json"
with open(all_genes_path, "w") as f:
    json.dump(all_genes, f, indent=2)

print(f"Saved genes list to {all_genes_path}")

all_techs_path = data_dir / "all_techs.json"
with open(all_techs_path, "w") as f:
    json.dump(techs, f, indent=2)

print(f"Saved {len(techs)} technologies list to {all_techs_path}")

# Cleanup: delete the original full dataset file 
del pancreas_adata  # drop reference to ensure no open handle
try:
    if os.path.exists(pancreas_adata_path):
        os.remove(pancreas_adata_path)
        print(f"Deleted '{pancreas_adata_path}'")
except Exception as e:
    print(f"[WARN] Could not delete '{pancreas_adata_path}': {e}")

# Distribute data to clients for federated learning
data_federated_dir = Path("data_federated")
data_shared_dir = data_federated_dir / "data_shared"
data_server_dir = data_federated_dir / "data_server"

data_federated_dir.mkdir(exist_ok=True)
data_server_dir.mkdir(exist_ok=True)

# For each tech, create client folder and save train/valid
train_techs = [t for t in techs if t not in ["smartseq2", "celseq2"]]

for i, tech in enumerate(train_techs):
    client_dir = data_federated_dir / f"data_client_{i}"
    client_dir.mkdir(exist_ok=True)

    train_client = pancreas_train[pancreas_train.obs["tech"] == tech]
    valid_client = pancreas_valid[pancreas_valid.obs["tech"] == tech]

    train_client.write(client_dir / "train.h5ad")
    valid_client.write(client_dir / "valid.h5ad")

    print(f"Client {i} ({tech}): train {train_client.shape}, valid {valid_client.shape}")

    all_genes_path = client_dir / "all_genes.json"
    with open(all_genes_path, "w") as f:
        json.dump(all_genes, f, indent=2)

    print(f"Saved genes list to {all_genes_path}")

    all_techs_path = client_dir / "all_techs.json"
    with open(all_techs_path, "w") as f:
        json.dump(techs, f, indent=2)

    print(f"Saved technologies list to {all_techs_path}")

all_genes_path = data_server_dir / "all_genes.json"
with open(all_genes_path, "w") as f:
    json.dump(all_genes, f, indent=2)

print(f"Saved genes list to {all_genes_path}")

all_techs_path = data_server_dir / "all_techs.json"
with open(all_techs_path, "w") as f:
    json.dump(techs, f, indent=2)

print(f"Saved technologies list to {all_techs_path}")