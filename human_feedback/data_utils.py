from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "training_runs" / "cnn1d_baseline"
PREDICTION_ARTIFACT = TRAINING_OUTPUT_DIR / "cnn1d_test_predictions.npz"
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
    label_path = data_dir / "activity_labels.txt"
    labels = {}

    with label_path.open("r", encoding="utf-8") as label_file:
        for line in label_file:
            raw_id, name = line.strip().split(maxsplit=1)
            labels[int(raw_id) - 1] = name

    return [labels[index] for index in range(len(labels))]


def load_test_signals(data_dir=DATA_DIR):
    signal_dir = data_dir / "test" / "Inertial Signals"
    signals = [
        np.loadtxt(signal_dir / f"{channel}_test.txt")
        for channel in CHANNEL_NAMES
    ]

    return np.transpose(np.array(signals), (1, 0, 2))


def load_prediction_artifact(path=PREDICTION_ARTIFACT):
    with np.load(path, allow_pickle=False) as artifact:
        data = {key: artifact[key] for key in artifact.files}

    required = {"test_index", "y_true", "y_pred"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Prediction artifact is missing: {sorted(missing)}")

    return data


def prediction_confidence(predictions):
    probabilities = predictions.get("probabilities")
    if probabilities is None:
        return None

    y_pred = predictions["y_pred"].astype(int)
    return probabilities[np.arange(len(y_pred)), y_pred]


def validate_prediction_order(predictions, x_test):
    test_index = predictions["test_index"]
    if len(test_index) != len(x_test):
        raise ValueError("Prediction count does not match X_test count.")

    expected = np.arange(len(x_test))
    if not np.array_equal(test_index, expected):
        raise ValueError("test_index must preserve the original X_test order.")


def confusion_matrix_counts(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for truth, prediction in zip(y_true, y_pred):
        matrix[truth, prediction] += 1

    return matrix


def indices_for_confusion_cell(y_true, y_pred, test_index, truth, prediction):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    test_index = np.asarray(test_index, dtype=int)
    mask = (y_true == truth) & (y_pred == prediction)

    return test_index[mask]


def overall_metrics(y_true, y_pred, class_names):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = np.arange(len(class_names))

    per_class_rows = []
    precisions = []
    recalls = []
    f1_scores = []

    for label, class_name in zip(labels, class_names):
        true_positive = int(((y_true == label) & (y_pred == label)).sum())
        predicted_positive = int((y_pred == label).sum())
        actual_positive = int((y_true == label).sum())

        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

        per_class_rows.append(
            {
                "Class": class_name,
                "Precision": precision,
                "Recall": recall,
                "F1": f1_score,
                "Support": actual_positive,
            }
        )

    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "per_class": per_class_rows,
    }


