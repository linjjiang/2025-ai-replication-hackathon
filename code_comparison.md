## Code Comparison vs. epiwek/retrocueing_RNN

- **Loss function**: Implemented Eq. 2.1 explicitly (`eq21_loss`) with circular distance weighting in PyTorch. Reference repo uses a custom loss in `retrocue_model.py` that multiplies MSE by circular distance; behavior aligned but our version is minimal and documented.
- **Initialization**: Orthogonal `W_hh`, Xavier `W_ih`/`W_out` as per paper; matches reference where recurrent weights use orthogonal init via `retrocue_model.py`.
- **Dataset/task**: Deterministic 512-trial grid with von Mises inputs, 7/7 delays, cue one-hot; mirrors `generate_data_von_mises.py` logic but implemented directly in `task.py` for clarity.
- **Training loop**: Our `scripts/train.py` runs RMSprop batch-size=1, noise injected via forward param, plateau/stop checks per Methods; reference uses CLI wrappers and evaluation looper but same optimizer/hyperparameters.
- **Geometry analysis**: Plane fitting via SVD with vertex-order orientation fix, AI via symmetric projection with clipping; reference `rep_geom_analysis.py` and `plane_angles_analysis.py` use PCA then cross-product normals with correction. We match intent but simplify and ensure numerical clipping.
- **Stress test**: We sweep post-cue delays with multi-run seeds inside `stress_test.py`; reference uses constants files (`constants_expt2_delay*`) and evaluation loops; functionality equivalent.
- **Logging**: Added structured `replication_log.txt` and JSON summaries; reference uses CSV/Mat files and mixture-model outputs. Our logging is lighter and text-based.
- **Figure generation**: Inline Matplotlib for figs 2B/4A/4C/5B; reference uses `plotting_funcs.py` with seaborn/matplotlib; aesthetics aligned (markers, variance labels, polar plots).
- **Scope differences**: Omitted behavioral mixture model fits and probe variants; focused on cued geometry experiments per user brief. Multi-run count reduced to 3 vs. paper’s 30 for speed.

