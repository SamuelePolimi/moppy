import torch

from interfaces.decoder_pro_mp import DecoderProMP


class DecoderDeepProMP(DecoderProMP):
    def __init__(self, stroke):
        self.stroke = stroke

    def generate_configuration(self, z: torch.Tensor, time: float):
        raise NotImplementedError()
