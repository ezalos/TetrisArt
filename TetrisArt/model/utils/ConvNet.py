import torch
from torch import nn



class NoNet(nn.Module):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__()

    def forward(self, x):
        return x


class ConvNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 25,
        layers_channels: int = 32,
        output_channels: int = 2,
        input_D: int= 2,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.2,
        pooling: bool = False,
    ):

        super().__init__()

        conv = {
            2: nn.Conv2d,
            3: nn.Conv3d,
        }[input_D]
        bn = {
            2: nn.BatchNorm2d,
            3: nn.BatchNorm3d,
        }[input_D]
        mp = {
            2: nn.MaxPool2d,
            3: nn.MaxPool3d,
        }[input_D]
        if pooling == False:
            mp = NoNet

        self.layers = nn.ModuleList(
            [
                conv(input_channels, layers_channels, kernel_size=kernel_size, padding="same"),
                bn(layers_channels),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                mp(kernel_size=kernel_size),
            ]
        )

        for i in range(1, num_layers):
            l = conv(layers_channels, layers_channels, kernel_size=kernel_size, padding="same")
            b = bn(layers_channels)
            r = nn.ReLU()
            d = nn.Dropout(p=dropout)
            m = mp(kernel_size=kernel_size)
            self.layers.extend([l, r, b, d, m])


    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x
