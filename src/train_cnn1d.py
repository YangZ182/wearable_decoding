import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from data_loader import CHANNEL_NAMES, DATA_DIR, load_activity_labels, load_uci_signals
from model import ActivityCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_RUNS_DIR = PROJECT_ROOT / "outputs" / "training_runs"


def subject_split(subjects, seed):
    count = int(np.ceil(len(np.unique(subjects)) * 0.2))
    validation_subjects = np.random.default_rng(seed).permutation(np.unique(subjects))[:count]
    validation_mask = np.isin(subjects, validation_subjects)
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


def make_loader(x, y, shuffle=False, seed=42):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(
        dataset, batch_size=64, shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )


def evaluate(model, loader, class_names):
    model.eval()
    all_labels, all_predictions, all_probabilities = [], [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x_batch, y_batch in loader:
            outputs = model(x_batch)
            total_loss += criterion(outputs, y_batch).item()
            all_labels.append(y_batch.numpy())
            all_predictions.append(outputs.argmax(dim=1).numpy())
            all_probabilities.append(torch.softmax(outputs, dim=1).numpy())
    y_true = np.concatenate(all_labels)
    probabilities = np.concatenate(all_probabilities)
    y_pred = np.concatenate(all_predictions)
    report = classification_report(
        y_true, y_pred, labels=np.arange(len(class_names)),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    return {
        "loss": total_loss / len(y_true),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": report["macro avg"]["f1-score"],
        "classification_report": report,
    }, y_true, y_pred, probabilities


def save_predictions(path, result, prediction_run_id):
    _, y_true, y_pred, probabilities = result
    np.savez(
        path, test_index=np.arange(len(y_true)), y_true=y_true, y_pred=y_pred,
        probabilities=probabilities, prediction_run_id=prediction_run_id,
    )


def evaluate_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    class_names = load_activity_labels()
    if checkpoint["channel_names"] != CHANNEL_NAMES or checkpoint["class_names"] != class_names:
        raise ValueError("Checkpoint channel/class ordering does not match the dataset.")
    model = ActivityCNN(depth=checkpoint.get("depth", 2))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    x_test = load_uci_signals(DATA_DIR / "test" / "Inertial Signals", "test")
    y_test = np.loadtxt(DATA_DIR / "test" / "y_test.txt", dtype=int) - 1
    result = evaluate(model, make_loader(x_test, y_test), class_names)
    metrics, y_true, y_pred, _ = result
    output_dir = checkpoint_path.parent
    run_id = checkpoint["prediction_run_id"]
    save_predictions(output_dir / "cnn1d_test_predictions.npz", result, run_id)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output_dir / "confusion_matrix.csv")
    metrics.update({
        "prediction_run_id": run_id, "class_names": class_names,
        "channel_names": CHANNEL_NAMES, "depth": checkpoint.get("depth", 2),
        "best_epoch": checkpoint.get("best_epoch"),
    })
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="CPU 1D-CNN training with subject-held-out validation.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--depth", type=int, choices=(2, 3), default=2)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.evaluate_checkpoint:
        evaluate_checkpoint(args.evaluate_checkpoint)
        return
    if args.epochs < 1 or args.weight_decay < 0:
        parser.error("epochs must be positive and weight-decay must be nonnegative")

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    x = load_uci_signals(DATA_DIR / "train" / "Inertial Signals", "train")
    y = np.loadtxt(DATA_DIR / "train" / "y_train.txt", dtype=int) - 1
    subjects = np.loadtxt(DATA_DIR / "train" / "subject_train.txt", dtype=int)
    train_indices, validation_indices = subject_split(subjects, seed=42)
    class_names = load_activity_labels()
    train_loader = make_loader(x[train_indices], y[train_indices], shuffle=True, seed=args.seed)
    validation_loader = make_loader(x[validation_indices], y[validation_indices])

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = TRAINING_RUNS_DIR / f"cnn1d_retrain_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "depth": args.depth, "epochs": args.epochs, "scheduler": args.scheduler,
        "weight_decay": args.weight_decay, "seed": args.seed,
        "learning_rate": 0.001, "batch_size": 64, "optimizer": "Adam",
    }
    run_info = {
        "prediction_run_id": run_id, "config": config,
        "train_subjects": np.unique(subjects[train_indices]).tolist(),
        "validation_subjects": np.unique(subjects[validation_indices]).tolist(),
        "train_count": len(train_indices), "validation_count": len(validation_indices),
        "selection": "validation macro_f1, then accuracy; earliest epoch on ties",
        "split_seed": 42,
        "python_executable": sys.executable, "python_version": sys.version,
        "torch_version": str(torch.__version__), "device": "cpu", "cpu_threads": 4,
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(f"Run: {output_dir}", flush=True)
    print(f"Train: {len(train_indices)}, validation: {len(validation_indices)}; {config}", flush=True)

    model = ActivityCNN(depth=args.depth)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, min_lr=1e-5,
    ) if args.scheduler else None
    training_log = []
    best_score = (-1.0, -1.0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        learning_rate = optimizer.param_groups[0]["lr"]
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
        validation, _, _, _ = evaluate(model, validation_loader, class_names)
        training_log.append({
            "epoch": epoch, "loss": total_loss / len(train_indices),
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"], "learning_rate": learning_rate,
        })
        score = (validation["macro_f1"], validation["accuracy"])
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if scheduler:
            scheduler.step(validation["loss"])
        print(
            f"Epoch {epoch}/{args.epochs}: loss={training_log[-1]['loss']:.4f}, "
            f"val_accuracy={validation['accuracy']:.4f}, val_macro_f1={validation['macro_f1']:.4f}, "
            f"lr={learning_rate:g}", flush=True,
        )
        pd.DataFrame(training_log).to_csv(output_dir / "training_log.csv", index=False)

    model.load_state_dict(best_state)
    result = evaluate(model, validation_loader, class_names)
    np.savez(
        output_dir / "validation_predictions.npz", train_index=validation_indices,
        y_true=result[1], y_pred=result[2], probabilities=result[3], prediction_run_id=run_id,
    )
    torch.save({
        "model_state_dict": best_state, "class_names": class_names,
        "channel_names": CHANNEL_NAMES, "prediction_run_id": run_id,
        "depth": args.depth, "best_epoch": best_epoch, "config": config,
        "validation_metrics": result[0],
    }, output_dir / "model_state_dict.pt")
    run_info.update({"best_epoch": best_epoch, "validation_metrics": result[0]})
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(f"Best epoch: {best_epoch}, validation Macro-F1: {best_score[0]:.4f}", flush=True)
    print("Test set was not evaluated. Use --evaluate-checkpoint after selecting a configuration.", flush=True)


if __name__ == "__main__":
    main()
