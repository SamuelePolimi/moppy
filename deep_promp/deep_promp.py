import torch

from deep_promp.encoder_deep_pro_mp import EncoderDeepProMP
from deep_promp.decoder_deep_pro_mp import DecoderDeepProMP
from interfaces.pro_mp import ProMPInterface
from trajectory.trajectory import Trajectory


class DeepProMPException(Exception):
    pass


class DeepProMP(ProMPInterface):

    def __init__(self, name, encoder: EncoderDeepProMP = None, decoder: DecoderDeepProMP = None):
        super().__init__(name, encoder, decoder)
        self.name = name
        self.encoder: EncoderDeepProMP = encoder

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

    def get_traj_distribution_at(self, time: float, trajectory: Trajectory):
        pass

    def encode(self):
        self.encode()

    def decode(self):
        self.decode()

    def train(self, trajectories):
        pass

    def test(self):
        pass

    def validate(self):
        pass

    def __str__(self):
        return f"MP: {self.name}"
