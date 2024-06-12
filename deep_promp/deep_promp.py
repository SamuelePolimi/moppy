import torch

from deep_promp.encoder_deep_pro_mp import EncoderDeepProMP
from deep_promp.decoder_deep_pro_mp import DecoderDeepProMP
from interfaces.pro_mp import ProMPInterface
from trajectory.trajectory import Trajectory


class DeepProMPException(Exception):
    pass


class DeepProMP(ProMPInterface):

    def __init__(self, name, encoder: EncoderDeepProMP, decoder: DecoderDeepProMP):
        super().__init__(name, encoder, decoder)

        if decoder is None:
            raise DeepProMPException("The decoder cannot be None.")
        if encoder is None:
            raise DeepProMPException("The encoder cannot be None.")

        #self.last_layer_size = self.encoder[-1].out_features
        #self.first_layer_size = self.decoder[0].in_features

        #print(self.last_layer_size)
        #print(self.first_layer_size)
        #if self.last_layer_size != self.first_layer_size:
        #    raise DeepProMPException("The last layer of the encoder should have the same size as the first layer of the decoder.")

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
