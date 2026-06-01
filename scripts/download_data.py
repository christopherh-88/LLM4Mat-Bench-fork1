"""
Download LLM4Mat-Bench dataset splits from Google Drive.
Works locally (for test/val) and on Kaggle (for large train splits).

Usage:
    python scripts/download_data.py --datasets mp snumat jarvis_dft --splits train test val
    python scripts/download_data.py --datasets mp --splits train  # Kaggle only (1.5 GB)
"""
import re, requests, os, argparse

# Google Drive file IDs for each dataset split
FILE_IDS = {
    "mp": {
        "train": "1jlhipDUbXflDK_AquMH1-8znHNf_r6CF",       # 1.5 GB
        "test": "1iWLRTxZtPA_wGMbk060D8XsAnnEt27oo",         # 122 MB
        "validation": "1MHQhPIH_0PFrkwq7zdBaI9w4n0AcCyvn",  # 121 MB
    },
    "snumat": {
        "train": "1YTULBvnkjSWdVhRlwos6oRZbwYPARtSl",        # 115 MB
        "test": "1z1crenrlZfhKonYX2GMm0d0sQC_WoMQP",
        "validation": "1Xfz35busd1EFzn2k2TibspHzWtYvskSi",
    },
    "jarvis_dft": {
        "train": "17oXUWE1fTC4GBb-O56f_M8JG4foSB-ci",        # 614 MB
        "test": "1zyJjS7SGlc_GJaAnxVC_kBIVNs-zsgz7",
        "validation": "1WSTUAmEyNlYwDnqCKT2SIODYm_Mh8DR7",
    },
}

def gdrive_download(session, file_id, dest_path, headers):
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

    name = os.path.basename(dest_path)
    print(f"  Downloading {name}...")
    with session.get(dl_url, headers=headers, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r    {done/1e6:.0f}/{total/1e6:.0f} MB ({100*done//total}%)", end="", flush=True)
    print(f"\n  Done: {dest_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mp"], choices=list(FILE_IDS))
    parser.add_argument("--splits", nargs="+", default=["test", "validation"],
                        choices=["train", "test", "validation"])
    parser.add_argument("--data_path", default="data/")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

    for dataset in args.datasets:
        dataset_dir = os.path.join(args.data_path, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        print(f"\n[{dataset}]")
        for split in args.splits:
            file_id = FILE_IDS[dataset][split]
            dest = os.path.join(dataset_dir, f"{split}.csv")
            if os.path.exists(dest) and not args.overwrite:
                size_mb = os.path.getsize(dest) / 1e6
                print(f"  {split}.csv already exists ({size_mb:.0f} MB), skipping. Use --overwrite to re-download.")
                continue
            gdrive_download(session, file_id, dest, headers)

if __name__ == "__main__":
    main()
