"""
Kaggle/Colab setup script for LLM4Mat-Bench.
Run this at the start of a Kaggle notebook to download train.csv and install deps.

On Kaggle T4: run each cell in a notebook, or !python scripts/kaggle_setup.py
"""
import subprocess, sys, os, re, requests

# ── 1. Install dependencies ───────────────────────────────────────────────────
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "numpy<2.0", "pandas==2.0.1", "transformers==4.23.1",
    "sentencepiece==0.1.97", "tokenizers==0.13.1",
    "torchmetrics==0.11.4", "scikit-learn==1.2.2",
    "huggingface_hub", "accelerate", "tqdm",
])
print("Dependencies installed.")

# ── 2. Download MP train.csv (1.5 GB) ─────────────────────────────────────────
FILE_IDS = {
    "mp/train.csv": "1jlhipDUbXflDK_AquMH1-8znHNf_r6CF",
    "mp/test.csv": "1iWLRTxZtPA_wGMbk060D8XsAnnEt27oo",
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
    uuid_m = re.search(r'name="uuid"\s+value="([^"]+)"', resp.text)
    if confirm and uuid_m:
        dl_url = (
            f"https://drive.usercontent.google.com/download"
            f"?id={file_id}&export=download&authuser=0"
            f"&confirm={confirm.group(1)}&uuid={uuid_m.group(1)}"
        )
    else:
        dl_url = url
    print(f"Downloading {dest_path}...")
    with session.get(dl_url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done/1e9:.2f}/{total/1e9:.2f} GB ({100*done//total}%)", end="", flush=True)
    print(f"\nDone: {dest_path} ({os.path.getsize(dest_path)/1e6:.0f} MB)")

DATA_DIR = "data"
for rel_path, file_id in FILE_IDS.items():
    gdrive_download(file_id, os.path.join(DATA_DIR, rel_path))

# ── 3. Set HF token from Kaggle secret ────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
    print("HF_TOKEN loaded from Kaggle secrets.")
except Exception:
    print("Note: HF_TOKEN not set. Add it to Kaggle secrets for gated models (Llama 2).")

print("\nSetup complete. Data is in data/mp/")
