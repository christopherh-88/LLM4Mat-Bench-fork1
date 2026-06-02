"""
Upload large data files and checkpoints to Hugging Face.

Steps before running:
  1. pip install huggingface_hub
  2. huggingface-cli login   (paste your HF token from https://huggingface.co/settings/tokens)
  3. python scripts/upload_to_hf.py
"""

from huggingface_hub import HfApi, create_repo
import os

HF_USERNAME = "christopherh-88"
DATASET_REPO = f"{HF_USERNAME}/LLM4Mat-Bench-data"
MODEL_REPO = f"{HF_USERNAME}/LLM4Mat-Bench-checkpoints"

DATA_FILES = [
    ("data/mp/train.csv",          "data/mp/train.csv"),
    ("data/mp/test.csv",           "data/mp/test.csv"),
    ("data/mp/validation.csv",     "data/mp/validation.csv"),
    ("data/jarvis_dft/train.csv",  "data/jarvis_dft/train.csv"),
    ("data/snumat/train.csv",      "data/snumat/train.csv"),
]

api = HfApi()

# Create repos if they don't exist
for repo_id, repo_type in [(DATASET_REPO, "dataset"), (MODEL_REPO, "model")]:
    try:
        create_repo(repo_id, repo_type=repo_type, exist_ok=True)
        print(f"Repo ready: {repo_id}")
    except Exception as e:
        print(f"Could not create {repo_id}: {e}")

# Upload data files
print("\nUploading data files...")
for local_path, hf_path in DATA_FILES:
    if not os.path.exists(local_path):
        print(f"  SKIP (not found): {local_path}")
        continue
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"  Uploading {local_path} ({size_mb:.0f} MB)...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=hf_path,
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"  Done: {hf_path}")

# Upload checkpoints
checkpoint_dir = "checkpoints"
if os.path.exists(checkpoint_dir):
    print("\nUploading checkpoints...")
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    print("  Done: all checkpoints uploaded")
else:
    print("\nNo checkpoints/ directory found, skipping.")

print(f"\nAll done!")
print(f"  Data:        https://huggingface.co/datasets/{DATASET_REPO}")
print(f"  Checkpoints: https://huggingface.co/{MODEL_REPO}")
