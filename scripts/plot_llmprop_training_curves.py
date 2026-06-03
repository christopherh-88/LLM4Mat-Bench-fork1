import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results" / "mp"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

def parse_val_mae(val_str):
    match = re.search(r"[\d]+\.[\d]+", str(val_str))
    return float(match.group()) if match else float(val_str)

def load_training_csv(path):
    df = pd.read_csv(path)
    df["validation mae loss"] = df["validation mae loss"].apply(parse_val_mae)
    return df

inputs = {
    "description": "llmprop_training_stats_for_band_gap_regression_description_no_stopwords_and_lengths_and_angles_replaced.csv",
    "formula":     "llmprop_training_stats_for_band_gap_regression_formula_no_stopwords_and_lengths_and_angles_replaced.csv",
}

test_results = {
    "description": {"mae": 0.5027, "mad_mae": 2.37, "r2": 0.570},
    "formula":     {"mae": 0.5502, "mad_mae": 2.16, "r2": 0.503},
}

colors = {"train": "#2c7bb6", "val": "#d7191c"}

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, height_ratios=[2.2, 1], hspace=0.55, wspace=0.35)

# --- Top row: training curves ---
for i, (input_type, filename) in enumerate(inputs.items()):
    df = load_training_csv(RESULTS_DIR / filename)
    ax = fig.add_subplot(gs[0, i])

    ax.plot(df["best_epoch"], df["training mae loss"],
            color=colors["train"], linewidth=1.8, label="Train MAE")
    ax.plot(df["best_epoch"], df["validation mae loss"],
            color=colors["val"], linewidth=1.8, linestyle="--", label="Val MAE")

    best_epoch = df.loc[df["validation mae loss"].idxmin(), "best_epoch"]
    best_val   = df["validation mae loss"].min()
    ax.axvline(best_epoch, color="gray", linewidth=1, linestyle=":", alpha=0.7)
    ax.scatter([best_epoch], [best_val], color=colors["val"], zorder=5, s=50)

    res = test_results[input_type]
    ax.set_title(
        f"Input: {input_type}\n"
        f"Test MAE={res['mae']:.4f} eV   MAD:MAE={res['mad_mae']:.2f}   R²={res['r2']:.3f}",
        fontsize=9, pad=8
    )
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("MAE Loss (eV)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(0, df["best_epoch"].max() + 3)
    ax.set_ylim(0, df["training mae loss"].max() * 1.1)
    ax.tick_params(labelsize=8)

    ax.annotate(
        f"Best epoch {int(best_epoch)}\nVal MAE={best_val:.4f}",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + df["best_epoch"].max() * 0.08, best_val + 0.08),
        fontsize=7.5, color="#444444",
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8)
    )

# --- Bottom row: results table (left) + deviations (right) ---
ax_table = fig.add_subplot(gs[1, 0])
ax_table.axis("off")

table_data = [
    ["Input",        "MAE (eV)", "MAD:MAE", "R²",   "Paper\nMAD:MAE"],
    ["description",  "0.5027",   "2.37",    "0.570", "≥ 9.1"],
    ["formula",      "0.5502",   "2.16",    "0.503", "≥ 9.1"],
]
tbl = ax_table.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.1, 2.0)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if row == 0:
        cell.set_facecolor("#2c7bb6")
        cell.set_text_props(color="white", fontweight="bold")
    elif col == 4:
        cell.set_facecolor("#fff3cd")
    elif col == 2:
        cell.set_facecolor("#fde8e8")
    else:
        cell.set_facecolor("#f9f9f9")
ax_table.set_title("Test Results vs. Paper (MP, band_gap)", fontsize=9, pad=10)

ax_dev = fig.add_subplot(gs[1, 1])
ax_dev.axis("off")
deviations = [
    ("max_len",          "888",   "256",   "OOM on Kaggle T4"),
    ("batch size",       "64",    "8",     "OOM on Kaggle T4"),
    ("train samples",    "125K",  "20K",   "OOM on Kaggle T4"),
    ("tokenizer",        "LLM-Prop", "T5", "Orig. link broken"),
]
col_labels = ["Parameter", "Paper", "Ours", "Reason"]
dev_tbl = ax_dev.table(
    cellText=deviations,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
dev_tbl.auto_set_font_size(False)
dev_tbl.set_fontsize(8)
dev_tbl.scale(1.1, 2.0)
for (row, col), cell in dev_tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if row == 0:
        cell.set_facecolor("#555555")
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#f9f9f9")
ax_dev.set_title("Deviations from Paper Config", fontsize=9, pad=10)

fig.suptitle(
    "Person B — LLM-Prop-35M Fine-tuning on Materials Project (band_gap)",
    fontsize=12, fontweight="bold", y=0.98
)

out = FIGURES_DIR / "person_B_llmprop_training_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.3)
print(f"Saved: {out}")
plt.show()
