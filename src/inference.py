from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_loader import CHANNEL_NAMES, load_activity_labels
from model import ActivityCNN


def load_model(checkpoint_path):
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
    if checkpoint["channel_names"] != CHANNEL_NAMES:
        raise ValueError("Checkpoint channel ordering does not match the data loader.")
    if checkpoint["class_names"] != load_activity_labels():
        raise ValueError("Checkpoint class ordering does not match activity_labels.txt.")
    model = ActivityCNN()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def make_test_loader(x_test, y_test):
    dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=64, shuffle=False)


def predict(model, test_loader):
    model.eval()
    all_predictions, all_labels, all_probabilities = [], [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch)
            all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            all_probabilities.append(torch.softmax(outputs, dim=1).cpu().numpy())
    return (
        np.array(all_labels), np.array(all_predictions),
        np.concatenate(all_probabilities, axis=0),
    )
