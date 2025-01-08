import torch
from torch import nn


class DQN1(nn.Module):
    """
    Deep Q-Network with 1 hidden layer, all layers are fully connected.
    """

    def __init__(self, input_shape: tuple[int, int], output_shape: int):
        super().__init__()
        self.fc1 = nn.Linear(input_shape[0] * input_shape[1], 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1) if x.dim() == 3 else x.view(1, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN2(nn.Module):
    """
    Deep Q-Network with CNN layers.
    """

    def __init__(self, input_shape: tuple[int, int], output_shape: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.size = input_shape[0] // 4 * input_shape[1] // 4
        self.fc1 = nn.Linear(32 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, output_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create(cnn: bool, input_shape: tuple[int, int], output_shape: int) -> nn.Module:
    return DQN2(input_shape, output_shape) if cnn else DQN1(input_shape, output_shape)
