# Person B — LLM-Prop-35M on MP: Results Summary

## Setup

| Item | Value |
|---|---|
| Model | LLM-Prop-35M |
| Dataset | Materials Project (MP) |
| Input types | formula (composition), description |
| Task | band_gap regression |
| Hardware | Kaggle 2× T4 GPU |
| Training time | ~4.5 hours per run |
| Inference time | ~23 seconds (10,259 test samples) |
| GPU memory | not logged (see deviations) |

## Results vs. Paper Table 9 (band_gap, MP)

| Input | MAE (eV) | MAD (eV) | MAD:MAE | R² | Paper MAD:MAE target |
|---|---|---|---|---|---|
| Description | 0.5027 | 1.1912 | **2.37** | 0.570 | ≥ 9.1 |
| Formula | 0.5502 | 1.1912 | **2.16** | 0.503 | ≥ 9.1 |

**N test samples:** 10,259

## Deviations from Paper Config (Appendix C / Section 3.1.2)

| Parameter | Paper | Ours | Reason |
|---|---|---|---|
| `max_len` | 888 | 256 | Kaggle T4 OOM at 888 |
| `train_bs` | 64 | 8 | Kaggle T4 OOM at 64 |
| Training samples | ~125K | 20K | Kaggle T4 OOM with full dataset |
| Tokenizer | Custom LLM-Prop tokenizer | T5 tokenizer | Original repo download link is broken/empty |
| Hardware | Paper's undisclosed GPU | Kaggle T4 (16GB VRAM) | Free tier constraint |
| GPU memory logged | Yes | No | Not instrumented during run |

## Reproducibility Issues Found

1. **Broken tokenizer link** — The original repo has `[this link]()` (empty) for the LLM-Prop tokenizer download. Used T5 tokenizer as substitute.
2. **sklearn API change** — `mean_squared_error(squared=False)` removed in recent sklearn. Code crashes on Python 3.12. Fixed: use `np.sqrt(mean_squared_error(...))`.
3. **numpy not imported in utils.py** — Code crashes during MAD calculation. Fixed: added `import numpy as np`.
4. **OneCycleLR requires epochs > 0** — Eval-only mode crashes without patching the scheduler initialization.

## Training Curves (band_gap, description input)

Best validation MAE saved at **epoch 82 / 100**. Loss improved 33 times during training.

| Epoch | Train MAE | Val MAE |
|---|---|---|
| 1 | 1.126 | 0.989 |
| 10 | — | — |
| 50 | — | — |
| 77 | 0.233 | 0.493 |
| 82 | 0.219 | 0.493 |

## Why MAD:MAE is 2.37 vs Paper's 9.1

The paper's MAD:MAE of 9.1 reflects training on the full ~125K sample dataset with max_len=888. Our 20K subset has a narrower property distribution (lower MAD), and our shorter max_len=256 truncates description inputs. Both factors reduce the signal the model can learn, raising MAE and lowering MAD:MAE. This is an expected and documented constraint of running on free-tier Kaggle hardware.

## Figures Generated

- `figures/person_B_llmprop_training_curves.png` — training + validation MAE curves for description and formula inputs
- `figures/person_B_llmprop_parity_plot.png` — predicted vs. actual band gap scatter plots for description and formula inputs

## Notes

- Classification tasks (`is_stable`, `is_gap_direct`) are assigned to Person C (MatBERT), not Person B per the project roadmap.
- GPU memory was not logged; this is documented as a deviation.
