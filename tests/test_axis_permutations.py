import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate_permutations import (
    CHECKPOINT, PERMUTATIONS, evaluate_permutations, permutation_channel_order,
    permute_imu_axes, save_results, summarize,
)
from inference import load_model, make_test_loader, predict


def test_axis_mapping():
    expected = {
        "xyz": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "xzy": [0, 2, 1, 3, 5, 4, 6, 8, 7],
        "yxz": [1, 0, 2, 4, 3, 5, 7, 6, 8],
        "yzx": [1, 2, 0, 4, 5, 3, 7, 8, 6],
        "zxy": [2, 0, 1, 5, 3, 4, 8, 6, 7],
        "zyx": [2, 1, 0, 5, 4, 3, 8, 7, 6],
    }
    x = np.arange(2 * 9 * 4, dtype=np.float32).reshape(2, 9, 4)
    for permutation, order in expected.items():
        assert permutation_channel_order(permutation) == order
        transformed = permute_imu_axes(x, permutation)
        assert transformed.shape == x.shape and transformed.dtype == x.dtype
        assert np.array_equal(transformed, x[:, order, :])
        assert not np.shares_memory(x, transformed)
        inverse = "".join("xyz"[permutation.index(axis)] for axis in "xyz")
        assert np.array_equal(permute_imu_axes(transformed, inverse), x)
    for invalid in ("xxz", "xy", "abc"):
        try:
            permute_imu_axes(x, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid permutation was accepted")
    for invalid_shape in (np.zeros((9, 128)), np.zeros((2, 128, 9)), np.zeros((2, 6, 128))):
        try:
            permute_imu_axes(invalid_shape, "xyz")
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid channel shape was accepted")


def test_summary_definition():
    table = pd.DataFrame({
        "permutation": PERMUTATIONS,
        "accuracy": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        "macro_f1": [0.8, 0.7, 0.3, 0.5, 0.4, 0.6],
    })
    summary = summarize(table)
    assert np.isclose(summary["mean_transformed_accuracy"], 0.6)
    assert np.isclose(summary["mean_transformed_macro_f1"], 0.5)
    assert summary["worst_case_accuracy"] == 0.4
    assert summary["worst_case_macro_f1"] == 0.3
    assert summary["worst_case_accuracy_permutation"] == "zyx"
    assert summary["worst_case_macro_f1_permutation"] == "yxz"
    assert np.isclose(summary["mean_accuracy_drop_pp"], 30)
    assert np.isclose(summary["worst_case_macro_f1_drop_pp"], 50)


def test_inference_and_saved_results():
    torch.set_num_threads(4)
    model, checkpoint = load_model(CHECKPOINT)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    x = np.random.default_rng(42).normal(size=(67, 9, 128))
    y = np.arange(67) % 6
    original_x, original_y = x.copy(), y.copy()
    x.flags.writeable = False
    y.flags.writeable = False
    class_names = checkpoint["class_names"]
    result = evaluate_permutations(model, x, y, class_names)
    table, predictions, probabilities, matrices = result
    assert tuple(table.permutation) == PERMUTATIONS
    assert predictions.shape == (6, 67) and probabilities.shape == (6, 67, 6)
    assert matrices.shape == (6, 6, 6)
    assert np.allclose(probabilities.sum(axis=2), 1)
    truth, baseline, baseline_probability = predict(model, make_test_loader(x, y))
    assert np.array_equal(truth, y)
    assert np.array_equal(predictions[0], baseline)
    assert np.array_equal(probabilities[0], baseline_probability)
    assert table.iloc[0].delta_accuracy_pp == table.iloc[0].delta_macro_f1_pp == 0
    for index in range(6):
        assert matrices[index].sum() == len(y)
        assert np.array_equal(matrices[index].sum(axis=1), np.bincount(y, minlength=6))
        assert np.isclose(np.trace(matrices[index]) / len(y), table.iloc[index].accuracy)
        assert np.isclose(table.iloc[index].macro_f1, f1_score(
            y, predictions[index], labels=np.arange(6), average="macro", zero_division=0,
        ))
        assert np.isclose(table.iloc[index].delta_accuracy_pp,
                          100 * (table.iloc[index].accuracy - table.iloc[0].accuracy))
    assert np.array_equal(x, original_x) and np.array_equal(y, original_y)
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert all(parameter.grad is None for parameter in model.parameters())
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "stress"
        save_results(output, result, y, class_names, {
            "model_run_id": checkpoint["prediction_run_id"], "checkpoint_sha256": "test",
            "input_shape": list(x.shape),
        })
        with np.load(output / "axis_permutations.npz", allow_pickle=False) as saved:
            for key in saved.files:
                assert saved[key].dtype != object
            assert np.array_equal(saved["y_pred"], predictions)
            assert np.array_equal(saved["y_true"], y)
            assert np.array_equal(saved["confusion_matrices"], matrices)
            assert np.array_equal(saved["test_index"], np.arange(len(y)))
            for column in table.columns[1:]:
                assert np.array_equal(saved[column], table[column])
        csv = pd.read_csv(output / "metrics.csv")
        assert np.allclose(csv.select_dtypes("number"), table.select_dtypes("number"))
        report = json.loads((output / "summary.json").read_text())
        assert report["results"] == table.to_dict(orient="records")
        assert report["summary"] == summarize(table)
        assert report["input_shape"] == [67, 9, 128]


if __name__ == "__main__":
    test_axis_mapping()
    test_summary_definition()
    test_inference_and_saved_results()
    print("axis permutation self-check passed")
