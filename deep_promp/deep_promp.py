from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from deep_promp.decoder_deep_pro_mp import DecoderDeepProMP
from deep_promp.encoder_deep_pro_mp import EncoderDeepProMP
from interfaces.movement_primitive import MovementPrimitive
from trajectory.trajectory import Trajectory


class DeepProMP(MovementPrimitive):
    """
    A DeepProMP is a probabilistic movement primitive that uses deep neural networks to encode and decode trajectories.
    """

    def __init__(self, name, encoder: EncoderDeepProMP, decoder: DecoderDeepProMP):
        super().__init__(name, encoder, decoder)
        self.encoder = encoder
        self.decoder = decoder
        if decoder is None or encoder is None:
            raise ValueError("The decoder or the encoder cannot be None.")

        # self.last_layer_size = self.encoder[-1].out_features
        # self.first_layer_size = self.decoder[0].in_features

        # print(self.last_layer_size)
        # print(self.first_layer_size)
        # if self.last_layer_size != self.first_layer_size:
        #    raise DeepProMPException("The last layer of the encoder should have the same size as the first layer of the decoder.")

    def train(self, trajectories: List[Trajectory]):

        # Loss function
        criterion = nn.MSELoss()

        # Optimizers
        optimizer = optim.Adam(list(self.encoder.net.parameters()) + list(self.decoder.net.parameters()), lr=0.001)
        for i in range(100):
            for data in trajectories:
                data: Trajectory = data
                target = data[0].to_vector()
                target = target[: 8]
                mu, sigma = self.encoder(data)
                latent_var_z = np.concatenate((np.array(mu), np.array(sigma)))
                losses = []
                for j in data.trajectory:
                    decoded = self.decoder(latent_var_z, j.time)
                    loss = criterion(decoded, torch.from_numpy(j.to_vector()[: 8]))
                    losses.append(loss)
                loss = sum(losses) / len(losses)
                print(loss)
                loss = loss.mean()
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def test(self):
        raise NotImplementedError()

    def validate(self):
        raise NotImplementedError()
