"""
Person B — Kaggle script for LLM-Prop classification runs.
Runs is_stable and is_gap_direct with formula + description inputs.
Logs GPU memory usage throughout.

Run in a Kaggle T4 notebook cell by cell, or as:
    !python scripts/kaggle_classification_runs.py
"""
import subprocess, sys, os, re, requests, torch

# ── 1. Install dependencies ───────────────────────────────────────────────────
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "numpy<2.0", "pandas==2.0.1", "transformers==4.23.1",
    "sentencepiece==0.1.97", "tokenizers==0.13.1",
    "torchmetrics==0.11.4", "scikit-learn==1.2.2",
    "huggingface_hub", "accelerate", "tqdm",
])

# ── 2. Clone repo ─────────────────────────────────────────────────────────────
REPO = "/kaggle/working/LLM4Mat-Bench-fork1"
if not os.path.exists(REPO):
    subprocess.check_call(["git", "clone",
        "https://github.com/christopherh-88/LLM4Mat-Bench-fork1", REPO])
os.chdir(REPO)

# ── 3. Download MP data ───────────────────────────────────────────────────────
FILE_IDS = {
    "mp/train.csv":      "1jlhipDUbXflDK_AquMH1-8znHNf_r6CF",
    "mp/test.csv":       "1iWLRTxZtPA_wGMbk060D8XsAnnEt27oo",
    "mp/validation.csv": "1MHQhPIH_0PFrkwq7zdBaI9w4n0AcCyvn",
}

def gdrive_download(file_id, dest_path):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1_000_000:
        print(f"Already exists: {dest_path}")
        return
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    resp = session.get(url, headers=headers)
    confirm = re.search(r'name="confirm"\s+value="([^"]+)"', resp.text)
    uuid_m  = re.search(r'name="uuid"\s+value="([^"]+)"', resp.text)
    dl_url = (f"https://drive.usercontent.google.com/download"
              f"?id={file_id}&export=download&authuser=0"
              f"&confirm={confirm.group(1)}&uuid={uuid_m.group(1)}"
              if confirm and uuid_m else url)
    print(f"Downloading {dest_path}...")
    with session.get(dl_url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                f.write(chunk); done += len(chunk)
                if total:
                    print(f"\r  {done/1e9:.2f}/{total/1e9:.2f} GB ({100*done//total}%)", end="", flush=True)
    print(f"\nDone: {dest_path}")

for rel, fid in FILE_IDS.items():
    gdrive_download(fid, os.path.join("data", rel))

# ── 4. Subsample train to 20K (same as band_gap runs) ─────────────────────────
import pandas as pd, numpy as np
np.random.seed(42)
train_full = pd.read_csv("data/mp/train.csv")
print(f"Full train size: {len(train_full)}")
train_sub = train_full.sample(n=min(20000, len(train_full)), random_state=42)
train_sub.to_csv("data/mp/train.csv", index=False)
print(f"Saved {len(train_sub)} subset to train.csv")

# ── 5. GPU memory helper ───────────────────────────────────────────────────────
def log_gpu_memory(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.max_memory_allocated() / 1e9
        reserved  = torch.cuda.max_memory_reserved() / 1e9
        print(f"[GPU MEM @ {label}] max_allocated={allocated:.2f}GB  max_reserved={reserved:.2f}GB")
        torch.cuda.reset_peak_memory_stats()

# ── 6. Run classification experiments ─────────────────────────────────────────
RUNS = [
    ("is_stable",     "formula"),
    ("is_stable",     "description"),
    ("is_gap_direct", "formula"),
    ("is_gap_direct", "description"),
]

for property_name, input_type in RUNS:
    print(f"\n{'='*60}")
    print(f"  RUNNING: {property_name} | {input_type}")
    print(f"{'='*60}")

    torch.cuda.reset_peak_memory_stats()

    cmd = [
        sys.executable, "code/llmprop_and_matbert/train.py",
        "--data_path",        "data/",
        "--results_path",     "results/",
        "--checkpoints_path", "checkpoints/",
        "--model_name",       "llmprop",
        "--dataset_name",     "mp",
        "--input_type",       input_type,
        "--property_name",    property_name,
        "--max_len",          "256",
        "--epochs",           "100",
        "--train_batch_size", "8",
        "--valid_batch_size", "8",
    ]
    result = subprocess.run(cmd, capture_output=False)

    log_gpu_memory(f"{property_name}/{input_type} post-train")

    if result.returncode != 0:
        print(f"ERROR: {property_name}/{input_type} failed with exit code {result.returncode}")
    else:
        print(f"Done: {property_name}/{input_type}")

print("\nAll classification runs complete.")
print("Results saved to results/mp/")
