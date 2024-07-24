import os
import pytest
import torch
from moppy.deep_promp import EncoderDeepProMP
from moppy.trajectory.state import JointConfiguration
from moppy.trajectory.state.sinus_state import SinusState
from moppy.trajectory.trajectory import Trajectory


@pytest.fixture(scope='function')
def encoder():
    return EncoderDeepProMP(latent_variable_dimension=2,
                            hidden_neurons=[10, 10],
                            trajectory_state_class=JointConfiguration,)


@pytest.fixture(scope='function')
def trajectory():
    ret = Trajectory[JointConfiguration]()
    point = JointConfiguration(joint_positions=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
                               gripper_open=torch.tensor([1]),
                               time=torch.tensor([0.0]))
    ret.add_point(point)
    return ret


class TestEncoderDeepProMP:
    def test_init(self):
        encoder = EncoderDeepProMP(latent_variable_dimension=2,
                                   hidden_neurons=[10, 10],
                                   trajectory_state_class=JointConfiguration,)

        # normal values
        assert encoder.latent_variable_dimension == 2
        assert encoder.hidden_neurons == [10, 10]
        # 9 from the JointConfiguration state dimension + time dimension
        # 4 from 2 * latent_variable_dimension
        assert encoder.neurons == [9, 10, 10, 4]
        assert encoder.input_dimension == 9  # 2 from the latent variable dimension
        # wrong values
        with pytest.raises(ValueError):
            EncoderDeepProMP(latent_variable_dimension=0,
                             hidden_neurons=[],
                             trajectory_state_class=JointConfiguration,)
        with pytest.raises(TypeError):
            EncoderDeepProMP(latent_variable_dimension=2,
                             hidden_neurons=[10, 10],
                             trajectory_state_class=EncoderDeepProMP)

    def test_encode_from_trajectory_state(self, encoder: EncoderDeepProMP, trajectory: Trajectory[JointConfiguration]):
        # normal values
        latent_variable = encoder.encode_to_latent_variable(trajectory=trajectory)
        assert len(latent_variable) == 2
        assert len(latent_variable[0]) == 2
        assert len(latent_variable[1]) == 2

        # wrong values
        with pytest.raises(ValueError):
            encoder.encode_to_latent_variable(None)
        with pytest.raises(ValueError):
            encoder.encode_to_latent_variable(trajectory=Trajectory[JointConfiguration]())
        with pytest.raises(ValueError):
            # wrong trajectory state class
            test_traj = Trajectory[SinusState]()
            test_sin_point = SinusState(value=torch.tensor([1.0]), time=torch.tensor([0.0]))
            test_traj.add_point(test_sin_point)
            encoder.encode_to_latent_variable(trajectory=test_traj)

    def test_save_load_model(self, encoder: EncoderDeepProMP):
        encoder.save_model()
        encoder.load_model()
        assert encoder.net is not None
        assert encoder.net.state_dict() is not None
        # delte the file
        os.remove("encoder_deep_pro_mp.pth")

    def test_sample_latent_variables(self, encoder: EncoderDeepProMP):
        # normal values
        mu = torch.tensor([1.0, 2.0])
        sigma = torch.tensor([0.1, 0.2])
        latent_variable = encoder.sample_latent_variables(mu=mu, sigma=sigma)
        assert len(latent_variable[0]) == 2  # 2 samples because latent_variable_dimension = 2

    def test_sample_latent_variable(self, encoder: EncoderDeepProMP):
        mu = torch.tensor([1.0, 2.0])
        sigma = torch.tensor([0.1, 0.2])
        latent_variable = encoder.sample_latent_variable(mu=mu, sigma=sigma)
        assert len(latent_variable) == 2

    def test_bayesian_aggregation(self, encoder: EncoderDeepProMP):
        mu_points = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        sigma_points = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        mu_z, sigma_z_sq = encoder.bayesian_aggregation(mu_points=mu_points, sigma_points=sigma_points)
        assert len(mu_z) == 2
        assert len(sigma_z_sq) == 2

    def test_save_and_load_model(self, encoder: EncoderDeepProMP):
        encoder.save_model()
        encoder.load_model()
        assert encoder.net is not None
        assert encoder.net.state_dict() is not None
        os.remove("encoder_deep_pro_mp.pth")

    def test_forward(self, encoder: EncoderDeepProMP, trajectory: Trajectory[JointConfiguration]):
        """Same as the test_encode_from_trajectory_state test"""
        latent_variable = encoder(trajectory=trajectory)
        assert len(latent_variable) == 2
        assert len(latent_variable[0]) == 2
        assert len(latent_variable[1]) == 2
