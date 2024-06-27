from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.distributions import kl, Normal

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
        self.latent_variable_dimension = encoder.latent_variable_dimension
        self.prior = Normal(torch.zeros(self.latent_variable_dimension), torch.ones(self.latent_variable_dimension))
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
        losses_traj = []
        for i in range(10):
            for data in trajectories:
                data: Trajectory = data
                mu, sigma = self.encoder(data)
                latent_var_z = torch.cat((mu, sigma), dim=0)
                losses = []
                for j in data.get_points():
                    decoded = self.decoder(latent_var_z, j.get_time())
                    loss = self.calculate_elbo(j.to_vector(), decoded)
                    losses.append(loss)
                loss = torch.mean(torch.stack(losses))
                print(loss)
                losses_traj.append(loss)
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Extract values from tensors
        values = [t.item() for t in losses_traj]

        # Plotting
        plt.plot(values, marker='o')
        plt.title('Tensor Values')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)

        # Save the plot as an image file
        plt.savefig('tensor_values_plot.png')

    # https://github.com/tonyduan/variational-autoencoders/blob/master/src/blocks.py
    def calculate_elbo(self, data, decoded):
        pred_z = self.encoder(data)
        kl_div = kl.kl_divergence(pred_z, self.prior)
        rec_loss = torch.sum(decoded.log_prob(data), dim=1, keepdim=True)
        return -(rec_loss - kl_div).squeeze(dim=1)

    def test(self):
        raise NotImplementedError()

    def validate(self):
        raise NotImplementedError()
