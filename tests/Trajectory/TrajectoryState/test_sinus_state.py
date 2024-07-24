import pytest
import torch

from moppy.trajectory.state import SinusState


@pytest.fixture(scope="function")
def state():
    """Return a default SinusState. (We want it all to be tensors)"""
    return SinusState(value=torch.tensor([1.0]), time=torch.tensor([0.0]))


class TestSinusState:

    def test_to_vector_without_time(self, state):
        assert state.to_vector_without_time().tolist() == [1.0]

    def test_from_vector_without_time(self):
        # test normal usage
        state = SinusState.from_vector_without_time(torch.tensor([1.0]))
        assert state.value.tolist() == [1.0]
        assert state.time.tolist() == [0.0]

        # test exceptional usage
        with pytest.raises(ValueError):
            # to many values
            SinusState.from_vector_without_time(torch.tensor([1.0, 0.0]))
        with pytest.raises(ValueError):
            # to few values
            SinusState.from_vector_without_time(torch.tensor([1.0, 0.0, 0.0]))

    def test_from_vector(self):
        # test normal usage
        state = SinusState.from_vector(torch.tensor([1.0, 0.0]))
        assert state.value.tolist() == [1.0]
        assert state.time.tolist() == [0.0]

        # test exceptional usage
        with pytest.raises(ValueError):
            # to many values
            SinusState.from_vector(torch.tensor([1.0, 0.0, 0.0]))
        with pytest.raises(ValueError):
            # to few values
            SinusState.from_vector(torch.tensor([1.0]))

    def test_from_dict(self):
        # test normal usage
        state = SinusState.from_dict({'value': 1.0, 'time': 0.0})
        assert state.value.tolist() == [1.0]
        assert state.time.tolist() == [0.0]

        # test exceptional usage
        with pytest.raises(KeyError):
            SinusState.from_dict({'value': 1.0})
        with pytest.raises(KeyError):
            SinusState.from_dict({'time': 0.0})

    def test_get_dimensions(self):
        assert SinusState.get_dimensions() == 2

    def test_get_time_dimension(self):
        assert SinusState.get_time_dimension() == 1

    def test_init(self):
        states_to_test = [
            SinusState(value=1.0, time=0.0),
            SinusState(value=torch.tensor([1.0]), time=torch.tensor([0.0])),
            SinusState(value=torch.tensor([1.0]), time=0.0),
            SinusState(value=1.0, time=torch.tensor([0.0])),
        ]

        for state in states_to_test:
            assert isinstance(state.value, torch.Tensor)
            assert isinstance(state.time, torch.Tensor)
            assert state.value.tolist() == [1.0]
            assert state.time.tolist() == [0.0]
            assert state.value.dim() == 1
            assert state.time.dim() == 1
