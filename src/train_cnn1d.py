from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from data_loader import CHANNEL_NAMES, load_activity_labels, load_uci_har
from inference import make_test_loader, predict
from model import ActivityCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_RUNS_DIR = PROJECT_ROOT / "outputs" / "training_runs"


def main():
    prediction_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = TRAINING_RUNS_DIR / f"cnn1d_retrain_{prediction_run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_uci_har()
    print("Original shapes:")
    print("X_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", x_test.shape)
    print("y_test:", y_test.shape)

    y_train = y_train - 1
    y_test = y_test - 1

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = make_test_loader(x_test, y_test)

    model = ActivityCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    training_log = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        training_log.append({"epoch": epoch + 1, "loss": average_loss})
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")

    all_labels, all_predictions, all_probabilities = predict(model, test_loader)
    accuracy = float((all_predictions == all_labels).mean())
    class_names = load_activity_labels()
    print(f"Test Accuracy: {accuracy:.4f}")
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)
    report_text = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4,
    )
    print(report_text)

    np.savez(
        output_dir / "cnn1d_test_predictions.npz",
        test_index=np.arange(len(all_labels)),
        y_true=all_labels,
        y_pred=all_predictions,
        probabilities=all_probabilities,
        prediction_run_id=prediction_run_id,
    )
    pd.DataFrame(training_log).to_csv(output_dir / "training_log.csv", index=False)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output_dir / "confusion_matrix.csv")

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        digits=4,
        output_dict=True,
    )
    metrics = {
        "prediction_run_id": prediction_run_id,
        "accuracy": accuracy,
        "classification_report": report,
        "class_names": class_names,
        "channel_names": CHANNEL_NAMES,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "channel_names": CHANNEL_NAMES,
            "accuracy": accuracy,
            "prediction_run_id": prediction_run_id,
        },
        output_dir / "model_state_dict.pt",
    )
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()



