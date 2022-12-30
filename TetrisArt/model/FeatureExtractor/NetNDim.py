
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import gym
from torch import nn
from TetrisArt.model.utils.ConvNet import ConvNet
from TetrisArt.model.utils.LinearNet import LinearNet


def optimize_NetNDim(trial):
	return {
		"dropout": trial.suggest_float('dropout', 0, 1, log=False),
		"cnn_channels": int(trial.suggest_int('cnn_channels', 8, 64, log=True)),
		"cnn_layers": trial.suggest_int('cnn_layers', 1, 6, log=False),
		"linear_neurons": int(trial.suggest_int('linear_neurons', 8, 128, log=True)),
		"linear_layers": trial.suggest_int('linear_layers', 1, 6, log=False),
	}


class NetNDim(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, 
		observation_space: gym.spaces.Box, 
		features_dim: int = 64,
		dropout: float = 0.2,
		cnn_channels: int = 32,
		cnn_layers: int = 2,
		linear_neurons: int = 32,
		linear_layers: int = 2,
	):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        input_D = len(observation_space.shape) - 1
        self.cnn = nn.Sequential(
			ConvNet(
				input_channels=n_input_channels,
				layers_channels=cnn_channels,
				output_channels=cnn_channels,
				input_D=input_D,
				num_layers=cnn_layers,
				kernel_size=3,
				dropout=dropout,
				pooling=False,
			),
            # nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            observations = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(observations).shape[1]

        self.linear = nn.Sequential(
            LinearNet(
				input_size=n_flatten,
				output_size=features_dim,
				num_layers=linear_layers,
				layers_size=linear_neurons,
				dropout=dropout,
			),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
