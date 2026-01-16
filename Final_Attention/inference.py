# inference.py
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# import your classes from training code
from model import (
    MOFDataManager,
    MOFSetTransformerTrainer,
    device
)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def save_true_vs_pred(y_true, y_pred, out_path, title):
    lim_min = float(min(np.min(y_true), np.min(y_pred)))
    lim_max = float(max(np.max(y_true), np.max(y_pred)))
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

@torch.no_grad()
def run_split_inference(trainer: MOFSetTransformerTrainer, split: str):
    trainer.encoder_decoder.eval()
    if split == "train":
        files = trainer.data_manager.get_train_files()
    elif split == "test":
        files = trainer.data_manager.get_test_files()
    else:
        raise ValueError("split must be 'train' or 'test'")

    preds, targets = trainer.forward_pass(files)
    if preds is None or len(preds) == 0:
        raise RuntimeError(f"No predictions generated for {split} split.")
    y_pred = preds.detach().cpu().numpy().reshape(-1)
    y_true = targets.detach().cpu().numpy().reshape(-1)
    return y_true, y_pred, files

def compute_metrics(y_true, y_pred):
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return r2, mae, rmse

def load_model_for_inference(trainer: MOFSetTransformerTrainer, weights_path: str):
    # discover columns & shapes, init model
    trainer.determine_all_soap_columns()
    trainer._initialize_models()
    # load state dict (supports {'encoder_decoder': ...} or raw dict)
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get("encoder_decoder", ckpt)
    trainer.encoder_decoder.load_state_dict(state)
    trainer.encoder_decoder.to(device)
    trainer.encoder_decoder.eval()
    print(f"✓ Loaded weights from: {weights_path}")

def main(
    folder_path="../CIF_files",
    target_csv="../id_labels.csv",
    weights_path="optimized_models_fast100.pth",
    outdir="outputs"
):
    ensure_dir(outdir)
    print("=== Inference: SetTransformer on MOF dataset ===")

    # data manager with your split
    dm = MOFDataManager(
        folder_path=folder_path,
        target_csv=target_csv,
        val_size=0.0,
        test_size=0.2,
        random_state=42
    )

    # trainer shell (no training)
    trainer = MOFSetTransformerTrainer(
        data_manager=dm,
        aggregator_params={
            'num_seed_points': 1,
            'num_encoder_blocks': 1,
            'num_decoder_blocks': 1,
            'heads': 4,
            'dropout': 0.3,
        },
        loss_metric='mae'
    )

    load_model_for_inference(trainer, weights_path)

    # --- TRAIN ---
    y_true_tr, y_pred_tr, files_tr = run_split_inference(trainer, "train")
    r2_tr, mae_tr, rmse_tr = compute_metrics(y_true_tr, y_pred_tr)
    print(f"[TRAIN] R2={r2_tr:.4f}  MAE={mae_tr:.4f}  RMSE={rmse_tr:.4f}  N={len(y_true_tr)}")
    save_true_vs_pred(
        y_true_tr, y_pred_tr,
        os.path.join(outdir, "true_vs_pred_train.png"),
        f"True vs Predicted — TRAIN (R²={r2_tr:.2f})"
    )
    # CSV
    np.savetxt(os.path.join(outdir, "predictions_train.csv"),
               np.column_stack([y_true_tr, y_pred_tr]),
               delimiter=",", header="y_true,y_pred", comments="")

    # --- TEST ---
    y_true_te, y_pred_te, files_te = run_split_inference(trainer, "test")
    r2_te, mae_te, rmse_te = compute_metrics(y_true_te, y_pred_te)
    print(f"[TEST ] R2={r2_te:.4f}  MAE={mae_te:.4f}  RMSE={rmse_te:.4f}  N={len(y_true_te)}")
    save_true_vs_pred(
        y_true_te, y_pred_te,
        os.path.join(outdir, "true_vs_pred_test.png"),
        f"True vs Predicted — TEST (R²={r2_te:.2f})"
    )
    # CSV
    np.savetxt(os.path.join(outdir, "predictions_test.csv"),
               np.column_stack([y_true_te, y_pred_te]),
               delimiter=",", header="y_true,y_pred", comments="")

    # Combined plot
    lim_min = float(min(y_true_tr.min(), y_true_te.min(), y_pred_tr.min(), y_pred_te.min()))
    lim_max = float(max(y_true_tr.max(), y_true_te.max(), y_pred_tr.max(), y_pred_te.max()))
    plt.figure(figsize=(6,6))
    plt.scatter(y_true_tr, y_pred_tr, alpha=0.5, label="Train")
    plt.scatter(y_true_te, y_pred_te, alpha=0.7, label="Test")
    plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title("True vs Predicted — Combined")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "true_vs_pred_combined.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    # adjust paths if needed
    main(
        folder_path="../CIF_files",
        target_csv="../id_labels.csv",
        weights_path="optimized_models_fast100.pth",  # or "optimized_models.pth"
        outdir="outputs"
    )
