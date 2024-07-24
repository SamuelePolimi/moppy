import pytest
import torch

from moppy.trajectory.state import JointConfiguration


@pytest.fixture(scope="function")
def state():
    """Return a default JointConfiguration. (We want it all to be tensors)"""
    return JointConfiguration(joint_positions=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
                              gripper_open=torch.tensor([1]),
                              time=torch.tensor([0.0]))


class TestJointConfiguration:

    def test_to_vector_without_time(self, state):
        assert state.to_vector_without_time().tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0]

    def test_from_vector_without_time(self):
        # test normal usage
        state = JointConfiguration.from_vector_without_time(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0]))
        assert state.joint_positions.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        assert state.gripper_open.tolist() == 1.0
        assert state.time.tolist()

        # test exceptional usage
        with pytest.raises(ValueError):
            # to many values
            JointConfiguration.from_vector_without_time(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0, 0.0]))
            # to few values
            JointConfiguration.from_vector_without_time(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))

    def test_from_vector(self):
        # test normal usage
        state = JointConfiguration.from_vector(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0, 0.0]))
        assert state.joint_positions.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        assert state.gripper_open.tolist() == 1.0
        assert state.time.tolist() == [0.0]

        # test exceptional usage
        with pytest.raises(ValueError):
            # to many values
            JointConfiguration.from_vector(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0, 0.0, 0.0]))
            # to few values
            JointConfiguration.from_vector(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1.0]))

    def test_from_dict(self):
        # test normal usage
        state = JointConfiguration.from_dict({'joint_positions': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 'gripper_open': 1.0, 'time': 0.0})
        assert state.joint_positions.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        assert state.gripper_open.tolist() == 1.0
        assert state.time.tolist() == [0.0]

        # test exceptional usage
        with pytest.raises(KeyError):
            JointConfiguration.from_dict({'joint_positions': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 'gripper_open': 1.0})
            JointConfiguration.from_dict({'joint_positions': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 'time': 0.0})
            JointConfiguration.from_dict({'gripper_open': 1.0, 'time': 0.0})

    def test_get_dimensions(self):
        assert JointConfiguration.get_dimensions() == 9

    def test_get_time_dimension(self):
        assert JointConfiguration.get_time_dimension() == 1

    def test_init(self):
        states_to_test = [
            JointConfiguration(joint_positions=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], gripper_open=1.0, time=0.0),
            JointConfiguration(joint_positions=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), gripper_open=torch.tensor(1.0), time=torch.tensor([0.0])),
            JointConfiguration(joint_positions=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]), gripper_open=1.0, time=torch.tensor([0.0])),
            JointConfiguration(joint_positions=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], gripper_open=torch.tensor(1.0), time=0.0),
        ]

        for state in states_to_test:
            assert isinstance(state.joint_positions, torch.Tensor)
            assert isinstance(state.gripper_open, torch.Tensor)
            assert isinstance(state.time, torch.Tensor)
            assert state.joint_positions.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
            assert state.gripper_open.tolist() == 1.0
            assert state.time.tolist() == [0.0]
            assert state.joint_positions.dim() == 1
            assert state.gripper_open.dim() == 0
            assert state.time.dim() == 1
