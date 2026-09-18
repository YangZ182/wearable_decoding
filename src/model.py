import torch
import torch.nn as nn


class ActivityCNN(nn.Module):
    def __init__(self, depth=2):
        super().__init__()
        if depth not in (2, 3):
            raise ValueError("depth must be 2 or 3")
        self.depth = depth
        self.conv1 = nn.Conv1d(in_channels=9, out_channels=32, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        if depth == 3:
            self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_features=64, out_features=6)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        if self.depth == 3:
            x = self.relu(self.conv3(x))
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return self.fc(x)


if __name__ == "__main__":
    model = ActivityCNN()
    x = torch.randn(64, 9, 128)
    output = model(x)
    print(model)
    print("Input shape :", x.shape)
    print("Output shape:", output.shape)
