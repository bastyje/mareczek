import torch
from torch import nn


class DQN1(nn.Module):
    """
    Deep Q-Network with 1 hidden layer, all layers are fully connected.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(210 * 160, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 18)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self._preprocess(x)
        x = x.view(x.size(0), -1) if x.dim() == 3 else x.view(1, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # @staticmethod
    # def _preprocess(x: torch.Tensor) -> torch.Tensor:
    #     return x.squeeze(1) if x.dim() == 4 else x.squeeze(0)

