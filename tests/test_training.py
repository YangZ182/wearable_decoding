import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "human_feedback"))

from data_loader import load_activity_labels
from data_utils import overall_metrics
from model import ActivityCNN
from train_cnn1d import evaluate, make_loader, subject_split


def main():
    torch.set_num_threads(4)
    subjects = np.loadtxt(PROJECT_ROOT / "data/train/subject_train.txt", dtype=int)
    labels = np.loadtxt(PROJECT_ROOT / "data/train/y_train.txt", dtype=int) - 1
    train, validation = subject_split(subjects, 42)
    assert not np.intersect1d(subjects[train], subjects[validation]).size
    assert np.array_equal(np.sort(np.concatenate([train, validation])), np.arange(len(subjects)))
    assert len(np.unique(subjects[validation])) == 5
    assert np.array_equal(np.unique(labels[train]), np.arange(6))
    assert np.array_equal(np.unique(labels[validation]), np.arange(6))
    repeated = subject_split(subjects, 42)
    assert all(np.array_equal(a, b) for a, b in zip((train, validation), repeated))

    torch.manual_seed(42)
    x = np.random.default_rng(42).normal(size=(67, 9, 128))
    y = np.arange(67) % 6
    class_names = load_activity_labels()
    for depth in (2, 3):
        model = ActivityCNN(depth)
        batch = torch.tensor(x[:6], dtype=torch.float32)
        assert model(batch).shape == (6, 6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        loss = torch.nn.functional.cross_entropy(model(batch), torch.tensor(y[:6]))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        result = evaluate(model, make_loader(x, y), class_names)
        metrics, truth, prediction, probabilities = result
        assert np.array_equal(truth, y)  # Includes the last partial batch.
        assert probabilities.shape == (67, 6)
        assert np.allclose(probabilities.sum(axis=1), 1)
        reference = overall_metrics(truth, prediction, class_names)
        assert metrics["accuracy"] == reference["accuracy"]
        assert np.isclose(metrics["macro_f1"], reference["macro_f1"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"depth": depth, "model_state_dict": model.state_dict()}, path)
            saved = torch.load(path, map_location="cpu", weights_only=True)
            restored = ActivityCNN(saved["depth"])
            restored.load_state_dict(saved["model_state_dict"], strict=True)
            restored_result = evaluate(restored, make_loader(x, y), class_names)
            assert np.array_equal(restored_result[2], prediction)
            assert np.array_equal(restored_result[3], probabilities)

    legacy = torch.load(
        PROJECT_ROOT / "outputs/training_runs/cnn1d_retrain_20260911T110414Z/model_state_dict.pt",
        map_location="cpu", weights_only=True,
    )
    ActivityCNN().load_state_dict(legacy["model_state_dict"], strict=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, min_lr=1e-5,
    )
    for _ in range(5):
        scheduler.step(1.0)
    assert optimizer.param_groups[0]["lr"] == 0.0005
    print("training self-check passed")


if __name__ == "__main__":
    main()
