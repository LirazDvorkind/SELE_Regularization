# Model Training

Scripts for training the SELE score model. Supports d32 and d500 variants.

## Files

- `sele-score-model-training-script.py` — local training script (CPU, run as Current File)
- `sele_score_model_training.ipynb` — Colab notebook for GPU training - the most recent and updated version is in Google Drive folder "Thesis".
- `test-score-models.py` — visualizes and compares score gradients from trained checkpoints

## Trained checkpoints

Output to `Data/score_model/`. Default filenames match `NesterovHyperparams.model_path`:

- d500 → `sele_score_net_d500.pt`
- d32 → `sele_score_net_d32.pt`
