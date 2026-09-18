import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from data_loader import CHANNEL_NAMES, DATA_DIR, load_activity_labels, load_uci_signals
from inference import load_model, make_test_loader, predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "human_feedback"))
from data_utils import confusion_matrix_counts, overall_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / "outputs/training_runs/cnn1d_retrain_20260911T110414Z/model_state_dict.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs/axis_permutation_stress_test"
PERMUTATIONS = ("xyz", "xzy", "yxz", "yzx", "zxy", "zyx")


def permutation_channel_order(permutation):
    if permutation not in PERMUTATIONS:
        raise ValueError(f"permutation must be one of {PERMUTATIONS}")
    return [
        CHANNEL_NAMES.index(f"{group}_{axis}")
        for group in ("body_acc", "body_gyro", "total_acc")
        for axis in permutation
    ]


def permute_imu_axes(x_test, permutation):
    if x_test.ndim != 3 or x_test.shape[1] != len(CHANNEL_NAMES):
        raise ValueError("Expected signals with shape (N, 9, T).")
    return x_test[:, permutation_channel_order(permutation), :]


def evaluate_permutations(model, x_test, y_test, class_names):
    rows, predictions, probabilities, matrices = [], [], [], []
    for permutation in PERMUTATIONS:
        transformed = permute_imu_axes(x_test, permutation)
        truth, prediction, probability = predict(model, make_test_loader(transformed, y_test))
        if not np.array_equal(truth, y_test):
            raise ValueError("Inference changed label or sample ordering.")
        metrics = overall_metrics(truth, prediction, class_names)
        rows.append({
            "permutation": permutation, "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
        })
        predictions.append(prediction)
        probabilities.append(probability)
        matrices.append(confusion_matrix_counts(truth, prediction, len(class_names)))
    for row in rows:
        row["delta_accuracy_pp"] = 100 * (row["accuracy"] - rows[0]["accuracy"])
        row["delta_macro_f1_pp"] = 100 * (row["macro_f1"] - rows[0]["macro_f1"])
    return pd.DataFrame(rows), np.stack(predictions), np.stack(probabilities), np.stack(matrices)


def summarize(table):
    original, transformed = table.iloc[0], table.iloc[1:]
    summary = {"transformed_permutations": list(PERMUTATIONS[1:])}
    for metric in ("accuracy", "macro_f1"):
        mean = float(transformed[metric].mean())
        worst = float(transformed[metric].min())
        summary[f"mean_transformed_{metric}"] = mean
        summary[f"worst_case_{metric}"] = worst
        summary[f"worst_case_{metric}_permutation"] = str(
            transformed.loc[transformed[metric].idxmin(), "permutation"]
        )
        summary[f"mean_{metric}_drop_pp"] = 100 * (float(original[metric]) - mean)
        summary[f"worst_case_{metric}_drop_pp"] = 100 * (float(original[metric]) - worst)
    return summary


def save_results(output_dir, result, y_test, class_names, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    table, predictions, probabilities, matrices = result
    summary = summarize(table)
    np.savez(
        output_dir / "axis_permutations.npz",
        permutations=np.array(PERMUTATIONS),
        channel_orders=np.array([permutation_channel_order(p) for p in PERMUTATIONS]),
        test_index=np.arange(len(y_test)), y_true=y_test, y_pred=predictions,
        probabilities=probabilities, confusion_matrices=matrices,
        class_names=np.array(class_names), channel_names=np.array(CHANNEL_NAMES),
        **{column: table[column].to_numpy() for column in table.columns if column != "permutation"},
        **{key: value for key, value in summary.items() if key != "transformed_permutations"},
        model_run_id=metadata["model_run_id"], checkpoint_sha256=metadata["checkpoint_sha256"],
    )
    table.to_csv(output_dir / "metrics.csv", index=False)
    report = {
        **metadata, "class_names": class_names, "channel_names": CHANNEL_NAMES,
        "delta_definition": "100 * (permuted - xyz), percentage points",
        "drop_definition": "100 * (xyz - mean/worst), percentage points; excludes xyz",
        "perturbation": "Same axis permutation on body_acc, body_gyro and total_acc; no sign flips or arbitrary 3D rotations.",
        "results": table.to_dict(orient="records"), "summary": summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Offline six-axis-permutation evaluation; no training.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output-dir already exists; choose a new directory to preserve previous results")
    torch.set_num_threads(4)
    model, checkpoint = load_model(args.checkpoint)
    x_test = load_uci_signals(DATA_DIR / "test/Inertial Signals", "test")
    y_test = np.loadtxt(DATA_DIR / "test/y_test.txt", dtype=int) - 1
    class_names = load_activity_labels()
    result = evaluate_permutations(model, x_test, y_test, class_names)
    reference_path = args.checkpoint.parent / "cnn1d_test_predictions.npz"
    reference_matches = None
    if reference_path.exists():
        with np.load(reference_path, allow_pickle=False) as saved:
            reference_matches = bool(
                str(saved["prediction_run_id"]) == checkpoint["prediction_run_id"]
                and np.array_equal(saved["test_index"], np.arange(len(y_test)))
                and np.array_equal(saved["y_true"], y_test)
                and np.array_equal(saved["y_pred"], result[1][0])
            )
        if not reference_matches:
            raise ValueError("xyz predictions do not reproduce this checkpoint's saved test artifact.")
    metadata = {
        "model_run_id": checkpoint["prediction_run_id"],
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(y_test), "device": "cpu", "batch_size": 64,
        "input_shape": list(x_test.shape),
        "python_executable": sys.executable, "torch_version": str(torch.__version__),
        "xyz_matches_saved_predictions": reference_matches,
    }
    save_results(args.output_dir, result, y_test, class_names, metadata)
    print(result[0].to_string(index=False))
    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
