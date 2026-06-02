"""
Download large data files and checkpoints from Hugging Face.

Usage:
  pip install huggingface_hub
  python scripts/download_from_hf.py
"""

from huggingface_hub import hf_hub_download, snapshot_download
import os

HF_USERNAME = "christopherh-88"
DATASET_REPO = f"{HF_USERNAME}/LLM4Mat-Bench-data"
MODEL_REPO = f"{HF_USERNAME}/LLM4Mat-Bench-checkpoints"

DATA_FILES = [
    ("data/mp/train.csv",          "data/mp"),
    ("data/mp/test.csv",           "data/mp"),
    ("data/mp/validation.csv",     "data/mp"),
    ("data/jarvis_dft/train.csv",  "data/jarvis_dft"),
    ("data/snumat/train.csv",      "data/snumat"),
]

print("Downloading data files...")
for hf_path, local_dir in DATA_FILES:
    os.makedirs(local_dir, exist_ok=True)
    local_file = os.path.join(local_dir, os.path.basename(hf_path))
    if os.path.exists(local_file):
        print(f"  Already exists, skipping: {local_file}")
        continue
    print(f"  Downloading {hf_path}...")
    hf_hub_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        filename=hf_path,
        local_dir=".",
    )
    print(f"  Saved to {local_file}")

print("\nDownloading checkpoints...")
snapshot_download(
    repo_id=MODEL_REPO,
    repo_type="model",
    local_dir="checkpoints",
)
print("  Saved to checkpoints/")

print("\nAll done!")
