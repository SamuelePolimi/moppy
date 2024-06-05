import torch
import torch.nn as nn

from promp import ProMP
from trajectory import Trajectory


class DeepProMPException(Exception):
    pass


class DeepProMP(ProMP):

    def __init__(self, name, encoder: nn.Module = None, decoder: nn.Module = None):
        self.name = name
        self.encoder = encoder

        if encoder is None:
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(28 * 28, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 36),
                torch.nn.ReLU(),
                torch.nn.Linear(36, 18),
                torch.nn.ReLU(),
                torch.nn.Linear(18, 10)
            )
        self.decoder = decoder
        if decoder is None:
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(9, 18),
                torch.nn.ReLU(),
                torch.nn.Linear(18, 36),
                torch.nn.ReLU(),
                torch.nn.Linear(36, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 28 * 28),
                torch.nn.Sigmoid()
            )
        self.last_layer_size = self.encoder[-1].out_features
        self.first_layer_size = self.decoder[0].in_features

        print(self.last_layer_size)
        print(self.first_layer_size)
        if self.last_layer_size != self.first_layer_size:
            raise DeepProMPException("The last layer of the encoder should have the same size as the first layer of the decoder.")

    def get_traj_distribution_at(time: float, trajectory: Trajectory):
        pass

    def encode():
        pass

    def decode():
        pass

    def train():
        pass

    def test():
        pass

    def validate():
        pass

    def __str__(self):
        return f"MP: {self.name}"


a = DeepProMP("test")
