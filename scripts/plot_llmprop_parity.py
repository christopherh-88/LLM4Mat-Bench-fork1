import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error

RESULTS_DIR = Path(__file__).parent.parent / "results" / "mp"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

inputs = {
    "description": "llmprop_test_stats_for_band_gap_regression_description_no_stopwords_and_lengths_and_angles_replaced.csv",
    "formula":     "llmprop_test_stats_for_band_gap_regression_formula_no_stopwords_and_lengths_and_angles_replaced.csv",
}

colors = {"description": "#2c7bb6", "formula": "#d7191c"}

fig = plt.figure(figsize=(13, 9))
gs = gridspec.GridSpec(2, 2, height_ratios=[2.5, 1], hspace=0.5, wspace=0.35)

for i, (input_type, filename) in enumerate(inputs.items()):
    df = pd.read_csv(RESULTS_DIR / filename)
    actual    = df["actual_band_gap"].values
    predicted = df["predicted_band_gap"].values

    mae = mean_absolute_error(actual, predicted)
    r2  = r2_score(actual, predicted)
    mad = np.mean(np.abs(actual - np.mean(actual)))

    ax = fig.add_subplot(gs[0, i])
    ax.scatter(actual, predicted, alpha=0.12, s=4,
               color=colors[input_type], rasterized=True)

    lim = max(actual.max(), predicted.max()) * 1.07
    lim = max(lim, 0.5)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.2,
            label="Perfect prediction", zorder=5)

    ax.set_xlim(-0.1, lim)
    ax.set_ylim(-0.1, lim)
    ax.set_xlabel("Actual Band Gap (eV)", fontsize=10)
    ax.set_ylabel("Predicted Band Gap (eV)", fontsize=10)
    ax.set_title(f"Input: {input_type}", fontsize=11, pad=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(labelsize=9)
    ax.set_aspect("equal", adjustable="box")

    stats_text = (
        f"N = {len(actual):,}\n"
        f"MAE = {mae:.4f} eV\n"
        f"MAD = {mad:.4f} eV\n"
        f"MAD:MAE = {mad/mae:.2f}\n"
        f"R² = {r2:.3f}"
    )
    ax.text(0.97, 0.03, stats_text,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#bbbbbb", alpha=0.95))

# Bottom row: summary table (left) + deviations note (right)
ax_tbl = fig.add_subplot(gs[1, 0])
ax_tbl.axis("off")

rows = [
    ["description", "0.5027", "2.37", "0.570", "≥ 9.1"],
    ["formula",     "0.5502", "2.16", "0.503", "≥ 9.1"],
]
col_labels = ["Input", "MAE (eV)", "MAD:MAE", "R²", "Paper\nMAD:MAE"]
tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 2.2)
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
ax_tbl.set_title("Test Results vs. Paper (MP, band_gap)", fontsize=10, pad=12)

ax_note = fig.add_subplot(gs[1, 1])
ax_note.axis("off")
note = (
    "Deviations from paper config (Kaggle T4 constraints):\n\n"
    "  • max_len: 888 → 256  (OOM)\n"
    "  • batch size: 64 → 8  (OOM)\n"
    "  • train set: 125K → 20K  (OOM)\n"
    "  • tokenizer: LLM-Prop → T5  (orig. link broken)\n\n"
    "Paper target: MAD:MAE ≥ 9.1\n"
    "Lower MAD:MAE expected given resource constraints."
)
ax_note.text(0.05, 0.5, note, transform=ax_note.transAxes,
             fontsize=8.5, va="center", ha="left",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f5f5f5",
                       edgecolor="#cccccc"))

fig.suptitle(
    "Person B — LLM-Prop-35M: Predicted vs. Actual Band Gap (MP test set, N=10,259)",
    fontsize=12, fontweight="bold"
)

out = FIGURES_DIR / "person_B_llmprop_parity_plot.png"
plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.3)
print(f"Saved: {out}")
plt.show()
