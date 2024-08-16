import os
import pytest
import torch
from moppy.deep_promp import DecoderDeepProMP
from moppy.trajectory.state import JointConfiguration


@pytest.fixture(scope='function')
def decoder():
    return DecoderDeepProMP(latent_variable_dimension=2,
                            hidden_neurons=[10, 10],
                            trajectory_state_class=JointConfiguration,)


class TestDecoderDeepProMP:
    def test_init(self):
        decoder = DecoderDeepProMP(latent_variable_dimension=2,
                                   hidden_neurons=[10, 10],
                                   trajectory_state_class=JointConfiguration,)

        # normal values
        assert decoder.latent_variable_dimension == 2
        assert decoder.hidden_neurons == [10, 10]
        # 3 from the latent variable and 1 from the time
        # 8 from the JointConfiguration state dimension
        assert decoder.neurons == [3, 10, 10, 8]
        assert decoder.output_dimension == 8  # 8 from the JointConfiguration state dimension
        # wrong values
        with pytest.raises(ValueError):
            DecoderDeepProMP(latent_variable_dimension=0,
                             hidden_neurons=[],
                             trajectory_state_class=JointConfiguration,)
        with pytest.raises(TypeError):
            DecoderDeepProMP(latent_variable_dimension=2,
                             hidden_neurons=[10, 10],
                             trajectory_state_class=DecoderDeepProMP)

    def test_decode_from_latent_variable(self, decoder):
        latent_variable = torch.tensor([1, 2])
        time = torch.tensor([0.1])
        trajectory_state = decoder.decode_from_latent_variable(latent_variable, time)
        assert trajectory_state.shape == torch.Size([8])  # 8 from the JointConfiguration state dimension

    def test_save_load_model(self, decoder: DecoderDeepProMP):
        decoder.save_model()
        decoder.load_model()
        assert decoder.net is not None
        assert decoder.net.state_dict() is not None
        # delte the file
        os.remove("decoder_model_deep_pro_mp.pth")

    def test_forward(self, decoder):
        """Same as the test_decode_from_latent_variable test"""
        latent_variable = torch.tensor([1, 2])
        time = torch.tensor([0.1])
        trajectory_state = decoder(latent_variable, time)
        assert trajectory_state.shape == torch.Size([8])
