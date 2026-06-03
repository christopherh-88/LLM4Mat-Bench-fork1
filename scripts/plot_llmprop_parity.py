import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("Person B — LLM-Prop-35M Parity Plot: Predicted vs. Actual Band Gap (MP test set)",
             fontsize=11, fontweight="bold")

for ax, (input_type, filename) in zip(axes, inputs.items()):
    df = pd.read_csv(RESULTS_DIR / filename)
    actual    = df["actual_band_gap"].values
    predicted = df["predicted_band_gap"].values

    mae = mean_absolute_error(actual, predicted)
    r2  = r2_score(actual, predicted)
    mad = np.mean(np.abs(actual - np.mean(actual)))

    # Density-colored scatter (subsample for speed if needed)
    ax.scatter(actual, predicted, alpha=0.15, s=4, color=colors[input_type], rasterized=True)

    # Perfect prediction line
    lim = max(actual.max(), predicted.max()) * 1.05
    lim = max(lim, 0.5)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.2, label="Perfect prediction")

    stats_text = (
        f"N = {len(actual):,}\n"
        f"MAE = {mae:.4f} eV\n"
        f"MAD = {mad:.4f} eV\n"
        f"MAD:MAE = {mad/mae:.2f}\n"
        f"R² = {r2:.3f}\n\n"
        f"Paper target\nMAD:MAE ≥ 9.1"
    )
    ax.text(0.97, 0.03, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.9))

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual Band Gap (eV)", fontsize=10)
    ax.set_ylabel("Predicted Band Gap (eV)", fontsize=10)
    ax.set_title(f"Input: {input_type}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(labelsize=9)
    ax.set_aspect("equal", adjustable="box")

plt.tight_layout(rect=[0, 0, 1, 0.95])

out = FIGURES_DIR / "person_B_llmprop_parity_plot.png"
plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.2)
print(f"Saved: {out}")
plt.show()
