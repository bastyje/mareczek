import torch
from torch import nn


class DQN1(nn.Module):
    """
    Deep Q-Network with 1 hidden layer, all layers are fully connected.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, ram: bool):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.ram = ram

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1) if x.dim() == (2 if self.ram else 3) else x.view(1, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN2(nn.Module):
    """
    Deep Q-Network with CNN layers.
    """

    def __init__(self, input_shape: tuple[int, ...], output_shape: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        size = self.__conv2d_size(input_shape, 3, 0, 2)
        size = self.__max_pool2d_size(size, 2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        size = self.__conv2d_size(size, 3, 0, 2)
        x_size, y_size = self.__max_pool2d_size(size, 2, 2)
        self.fc1 = nn.Linear(32 * x_size * y_size, 256)
        self.fc2 = nn.Linear(256, output_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.dim() == 3
        x = torch.relu(self.conv1(x.unsqueeze(1 if batch else 0)))
        x = self.max_pool1(x)
        x = self.dropout1(x)
        x = torch.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1) if batch else x.view(1, -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    @staticmethod
    def __conv2d_size(value: tuple[int, ...], kernel: int, padding: int, stride: int) -> tuple[int, ...]:
        return tuple(map(lambda x: (x - kernel + 2 * padding) // stride + 1, value))

    @staticmethod
    def __max_pool2d_size(value: tuple[int, ...], kernel: int, stride: int) -> tuple[int, ...]:
        return tuple(map(lambda x: (x - kernel) // stride + 1, value))

