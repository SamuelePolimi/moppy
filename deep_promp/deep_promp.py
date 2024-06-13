import torch

from deep_promp.encoder_deep_pro_mp import EncoderDeepProMP
from deep_promp.decoder_deep_pro_mp import DecoderDeepProMP
from interfaces.movement_primitive import MovementPrimitive
from trajectory.trajectory import Trajectory


class DeepProMPException(Exception):
    pass


class DeepProMP(MovementPrimitive):

    def __init__(self, name, encoder: EncoderDeepProMP, decoder: DecoderDeepProMP):
        super().__init__(name, encoder, decoder)

        if decoder is None or encoder is None:
            raise ValueError("The decoder or the encoder cannot be None.")

        #self.last_layer_size = self.encoder[-1].out_features
        #self.first_layer_size = self.decoder[0].in_features

        #print(self.last_layer_size)
        #print(self.first_layer_size)
        #if self.last_layer_size != self.first_layer_size:
        #    raise DeepProMPException("The last layer of the encoder should have the same size as the first layer of the decoder.")

    def train(self, trajectories):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

    def validate(self):
        raise NotImplementedError()