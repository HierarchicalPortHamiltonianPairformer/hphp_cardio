"""
scripts/calibrate.py
Post-hoc isotonic regression calibration for HPHP-Cardio predictions.

Usage:
    python3 scripts/calibrate.py \
        --checkpoint checkpoints/stage2_xtb/best_model_fold_5.pt \
        --data       data/cipa_train.csv \
        --save       checkpoints/stage2_xtb/calibrator.pkl
"""
import argparse, pickle, sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, wilcoxon
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data",       default="data/cipa_train.csv")
    p.add_argument("--save",       default="checkpoints/stage2_xtb/calibrator.pkl")
    p.add_argument("--deepHERG-mae", type=float, default=14.7710,
                   help="Published DeepHERG MAE for Wilcoxon comparison")
    return p.parse_args()


def load_predictions(checkpoint_path, data_path):
    """
    Load raw model predictions from checkpoint.
    Expects checkpoint to contain 'cv_predictions' dict with keys:
        preds_raw : np.ndarray shape [28]
        targets   : np.ndarray shape [28]
        risk_class: np.ndarray shape [28]  (2=High, 1=Med, 0=Low)
    """
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "cv_predictions" not in ckpt:
        raise KeyError(
            "Checkpoint does not contain 'cv_predictions'. "
            "Re-run evaluate.py --save-predictions to generate."
        )

    cv = ckpt["cv_predictions"]
    preds_raw  = np.array(cv["preds_raw"])
    targets    = np.array(cv["targets"])
    risk_class = np.array(cv["risk_class"])

    df = pd.read_csv(data_path)
    assert len(df) == len(targets), \
        f"Data length mismatch: {len(df)} compounds vs {len(targets)} predictions"

    return preds_raw, targets, risk_class


def calibrate(preds_raw, targets, risk_class, deepHERG_mae):
    # Fit isotonic regression
    ir = IsotonicRegression(out_of_bounds="clip", increasing=True)
    ir.fit(preds_raw, targets)
    preds_cal = ir.transform(preds_raw)

    # Metrics
    rmse_raw = float(np.sqrt(np.mean((preds_raw - targets) ** 2)))
    rmse_cal = float(np.sqrt(np.mean((preds_cal - targets) ** 2)))
    mae_cal  = float(np.mean(np.abs(preds_cal - targets)))
    rho_cal  = float(spearmanr(preds_cal, targets).statistic)

    # AUROC: High vs Low+Med
    binary_labels = (risk_class >= 2).astype(int)
    auroc = float(roc_auc_score(binary_labels, preds_cal))

    # Calibration slope (true = slope * pred + intercept)
    slope, intercept = np.polyfit(preds_cal, targets, 1)
    r2 = float(np.corrcoef(preds_cal, targets)[0, 1] ** 2)

    # Wilcoxon vs DeepHERG
    errors_ours   = np.abs(preds_cal - targets)
    errors_dh     = np.full_like(errors_ours, deepHERG_mae)
    w_stat, p_val = wilcoxon(errors_ours, errors_dh, alternative="less")

    bonferroni_alpha = 0.0125
    significant = bool(p_val < bonferroni_alpha)

    return {
        "calibrator"     : ir,
        "rmse_raw_ms"    : rmse_raw,
        "rmse_cal_ms"    : rmse_cal,
        "mae_cal_ms"     : mae_cal,
        "spearman_rho"   : rho_cal,
        "auroc"          : auroc,
        "calib_slope"    : float(slope),
        "calib_intercept": float(intercept),
        "r2"             : r2,
        "wilcoxon_W"     : float(w_stat),
        "wilcoxon_p"     : float(p_val),
        "significant"    : significant,
        "deepHERG_mae"   : deepHERG_mae,
        "preds_cal"      : preds_cal,
        "targets"        : targets,
        "risk_class"     : risk_class,
    }


def print_report(res):
    sep = "─" * 52
    print(f"\n{sep}")
    print("  HPHP-Cardio Calibration Report")
    print(sep)
    print(f"  RMSE (uncalibrated)  : {res['rmse_raw_ms']:6.2f} ms")
    print(f"  RMSE (calibrated)    : {res['rmse_cal_ms']:6.2f} ms")
    print(f"  MAE  (calibrated)    : {res['mae_cal_ms']:6.2f} ms")
    print(f"  MAE  (DeepHERG)      : {res['deepHERG_mae']:6.2f} ms")
    print(f"  Spearman ρ           : {res['spearman_rho']:6.3f}")
    print(f"  AUROC (High vs rest) : {res['auroc']:6.3f}")
    print(f"  Calibration slope    : {res['calib_slope']:6.3f}  (target 0.9–1.1)")
    print(f"  R²                   : {res['r2']:6.3f}")
    print(sep)
    print(f"  Wilcoxon W={res['wilcoxon_W']:.0f},  p={res['wilcoxon_p']:.4e}")
    sig = "✓ SIGNIFICANT" if res["significant"] else "✗ NOT significant"
    print(f"  vs Bonferroni α=0.0125: {sig}")
    print(sep + "\n")


def main():
    args = parse_args()

    print("Loading predictions...")
    preds_raw, targets, risk_class = load_predictions(
        args.checkpoint, args.data
    )

    print("Fitting isotonic regression calibrator...")
    res = calibrate(preds_raw, targets, risk_class, args.deepHERG_mae)

    print_report(res)

    # Save calibrator
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({
            "calibrator" : res["calibrator"],
            "metrics"    : {k: v for k, v in res.items()
                            if k not in ("calibrator","preds_cal","targets","risk_class")},
        }, f)
    print(f"  Calibrator saved → {save_path}")

    # Also dump metrics as JSON for logging
    metrics_path = save_path.with_suffix(".json")
    metrics = {k: v for k, v in res.items()
               if k not in ("calibrator","preds_cal","targets","risk_class")}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved    → {metrics_path}\n")


if __name__ == "__main__":
    main()
