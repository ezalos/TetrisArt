import torch
from torch import nn


class LinearNet(nn.Module):
    def __init__(
        self,
        input_size: int = 25,
        output_size: int = 2,
        num_layers: int = 2,
        layers_size: int = 10,
        dropout: float = 0.2,
    ):

        super().__init__()

        self.linears = nn.ModuleList(
            [
                nn.Linear(input_size, layers_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
        )

        for i in range(1, num_layers - 1):
            l = nn.Linear(layers_size, layers_size)
            r = nn.ReLU()
            d = nn.Dropout(p=dropout)
            self.linears.extend([l, r, d])

        self.linears.append(nn.Linear(layers_size, output_size))
        self.linears.append(nn.ReLU())

    def forward(self, x):
        for l in self.linears:
            x = l(x)
        return x
