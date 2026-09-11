import sys
import tempfile
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[1] / "human_feedback"
sys.path.insert(0, str(MODULE_DIR))

from data_utils import (
    CHANNEL_NAMES,
    confusion_matrix_counts,
    indices_for_confusion_cell,
    overall_metrics,
    prediction_confidence,
    validate_prediction_order,
)
from review_utils import REVIEW_LABELS, load_reviews, review_for_sample, reviewed_error_summary, save_review
from visualization import matrix_cell_color, group_y_ranges, sample_signal_figure, signal_groups


def test_prediction_helpers():
    predictions = {
        "test_index": np.array([0, 1, 2, 3]),
        "y_true": np.array([0, 0, 1, 1]),
        "y_pred": np.array([0, 1, 1, 0]),
        "probabilities": np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
                [0.1, 0.9],
                [0.6, 0.4],
            ]
        ),
    }

    validate_prediction_order(predictions, np.zeros((4, 9, 128)))

    confidence = prediction_confidence(predictions)
    assert np.allclose(confidence, [0.8, 0.7, 0.9, 0.6])

    metrics = overall_metrics(
        predictions["y_true"],
        predictions["y_pred"],
        ["A", "B"],
    )
    assert metrics["accuracy"] == 0.5
    assert metrics["macro_precision"] == 0.5
    assert metrics["macro_recall"] == 0.5
    assert metrics["macro_f1"] == 0.5
    assert metrics["per_class"][0]["Support"] == 2

    matrix = confusion_matrix_counts(
        predictions["y_true"],
        predictions["y_pred"],
        num_classes=2,
    )
    assert np.array_equal(matrix, [[1, 1], [1, 1]])

    selected = indices_for_confusion_cell(
        predictions["y_true"],
        predictions["y_pred"],
        predictions["test_index"],
        truth=0,
        prediction=1,
    )
    assert np.array_equal(selected, [1])


def test_signal_plot_helpers():
    x_test = np.zeros((2, 9, 4))
    x_test[0, 0, :] = [-1.0, 0.0, 0.5, 1.0]
    x_test[1, 3, :] = [-2.0, 0.0, 1.0, 2.0]
    x_test[1, 8, :] = [0.2, 0.4, 0.6, 0.8]

    groups = signal_groups(CHANNEL_NAMES)
    assert [name for name, _ in groups] == [
        "body_acc_*（身体加速度）",
        "body_gyro_*（身体陀螺仪）",
        "total_acc_*（总加速度）",
    ]
    assert [len(indices) for _, indices in groups] == [3, 3, 3]

    ranges = group_y_ranges(x_test, groups)
    figure = sample_signal_figure(x_test[1], CHANNEL_NAMES, groups, ranges)
    assert len(figure.data) == 9

    assert matrix_cell_color(0, 4) == "#FFFFFF"
    assert matrix_cell_color(4, 4) == "#0C54AC"


def test_review_upsert():
    with tempfile.TemporaryDirectory() as temp_dir:
        review_path = Path(temp_dir) / "reviews.csv"

        save_review(
            review_path,
            prediction_run_id="run-1",
            test_index=31,
            truth="SITTING",
            prediction="STANDING",
            confidence=0.77,
            review_label=REVIEW_LABELS[0],
            review_note="first note",
        )
        save_review(
            review_path,
            prediction_run_id="run-1",
            test_index=31,
            truth="SITTING",
            prediction="STANDING",
            confidence=0.77,
            review_label=REVIEW_LABELS[1],
            review_note="updated note",
        )

        reviews = load_reviews(review_path)
        assert len(reviews) == 1

        review = review_for_sample(reviews, 31)
        assert review["review_label"] == REVIEW_LABELS[1]
        assert review["review_note"] == "updated note"

        save_review(
            review_path,
            prediction_run_id="run-1",
            test_index=305,
            truth="STANDING",
            prediction="SITTING",
            confidence=0.69,
            review_label=REVIEW_LABELS[1],
            review_note="second sample",
        )
        save_review(
            review_path,
            prediction_run_id="run-1",
            test_index=1,
            truth="WALKING",
            prediction="WALKING",
            confidence=0.99,
            review_label=REVIEW_LABELS[0],
            review_note="manual csv row that summary should ignore",
        )

        summary = reviewed_error_summary(load_reviews(review_path))
        assert summary["label_summary"]["count"].sum() == 2
        assert len(summary["pair_summary"]) == 2


if __name__ == "__main__":
    test_prediction_helpers()
    test_signal_plot_helpers()
    test_review_upsert()
    print("data_utils self-check passed")

