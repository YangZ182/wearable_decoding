from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

CHANNEL_NAMES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


def load_activity_labels(data_dir=DATA_DIR):
    labels_path = Path(data_dir) / "activity_labels.txt"
    labels = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            _, label = line.split(maxsplit=1)
            labels.append(label)
    return labels


def load_uci_signals(signal_dir, split):
    signals = [
        np.loadtxt(Path(signal_dir) / f"{channel}_{split}.txt")
        for channel in CHANNEL_NAMES
    ]
    return np.transpose(np.array(signals), (1, 0, 2))


def load_uci_har(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    x_train = load_uci_signals(train_dir / "Inertial Signals", "train")
    x_test = load_uci_signals(test_dir / "Inertial Signals", "test")
    y_train = np.loadtxt(train_dir / "y_train.txt", dtype=int)
    y_test = np.loadtxt(test_dir / "y_test.txt", dtype=int)

    return x_train, y_train, x_test, y_test
