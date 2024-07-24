import os
import pytest
import torch

from moppy.trajectory import Trajectory
from moppy.trajectory.state import SinusState, JointConfiguration


def get_absolute_path(relative_path):
    # Use __file__ to get the directory of the current file
    return os.path.abspath(os.path.join(os.path.dirname(__file__), relative_path))


@pytest.fixture(scope="function")
def trajectory_sinus_empty():
    return Trajectory[SinusState]()


@pytest.fixture(scope="function")
def trajectory_sinus():
    """Dummy sinus trajectory. Has 100 points. Time goes from 0 to 1.

    Note:
        All the used function should be tested in the test_state.py file.
    """
    num_steps = 100
    traj: Trajectory = Trajectory[SinusState]()
    t = 0.0
    for _ in range(num_steps):
        traj.add_point(SinusState(value=t, time=t))  # no need to any usefull value
        t += 1/(num_steps - 1)
    return traj


@pytest.fixture(scope="function")
def trajectory_joint_empty():
    return Trajectory[JointConfiguration]()


@pytest.fixture(scope="function")
def trajectory_joint():
    """Dummy joint trajectory. Has 100 points. Time goes from 0 to 1.

    Note:
        All the used function should be tested in the test_state.py file.
    """
    num_steps = 100
    traj: Trajectory = Trajectory[JointConfiguration]()
    t = 0.0
    for _ in range(num_steps):
        traj.add_point(JointConfiguration(joint_positions=[1, 2, 3, 4, 5, 6, 7], gripper_open=1, time=t))
        t += 1/(num_steps - 1)
    return traj


class TestTrajectory:

    def test_fixture(self, trajectory_sinus, trajectory_sinus_empty, trajectory_joint, trajectory_joint_empty):
        assert len(trajectory_sinus.get_points()) == 100
        assert len(trajectory_sinus_empty.get_points()) == 0
        assert len(trajectory_joint.get_points()) == 100
        assert len(trajectory_joint_empty.get_points()) == 0

    def test_add_point(self, trajectory_sinus_empty, trajectory_joint_empty):
        # Test normale usgage
        new_sin_point = SinusState(value=1.0, time=0.0)
        trajectory_sinus_empty.add_point(new_sin_point)
        assert len(trajectory_sinus_empty) == 1
        assert trajectory_sinus_empty.get_points()[0] == new_sin_point

        new_joint_point = JointConfiguration(joint_positions=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], gripper_open=1, time=0.0)
        trajectory_joint_empty.add_point(new_joint_point)
        assert len(trajectory_joint_empty) == 1
        assert trajectory_joint_empty.get_points()[0] == new_joint_point

        # Test exceptional usages
        new_joint_point = JointConfiguration(joint_positions=[1.0], gripper_open=True, time=0.0)
        with pytest.raises(ValueError):
            trajectory_sinus_empty.add_point(new_joint_point)

    def test_get_points(self, trajectory_sinus, trajectory_sinus_empty, trajectory_joint, trajectory_joint_empty):
        assert len(trajectory_sinus.get_points()) == 100
        assert len(trajectory_sinus_empty.get_points()) == 0
        assert len(trajectory_joint.get_points()) == 100
        assert len(trajectory_joint_empty.get_points()) == 0

    def test_get_times(self, trajectory_sinus, trajectory_sinus_empty, trajectory_joint, trajectory_joint_empty):
        assert trajectory_sinus.get_times() == [state.get_time() for state in trajectory_sinus.get_points()]
        assert trajectory_sinus_empty.get_times() == []
        assert trajectory_joint.get_times() == [state.get_time() for state in trajectory_joint.get_points()]
        assert trajectory_joint_empty.get_times() == []

    def test_to_vector(self, trajectory_sinus, trajectory_sinus_empty, trajectory_joint, trajectory_joint_empty):
        assert trajectory_sinus.to_vector().shape == torch.Size([100])
        assert trajectory_sinus_empty.to_vector().shape == torch.Size([0])
        assert trajectory_sinus.to_vector()[0] == trajectory_sinus.get_points()[0].to_vector_without_time()

        assert trajectory_joint.to_vector().shape == torch.Size([100*8])
        assert trajectory_joint_empty.to_vector().shape == torch.Size([0])
        assert trajectory_joint.to_vector()[0], trajectory_joint.get_points()[0].to_vector_without_time()

    def test_load_points_from_file(self):
        # Test normal usage
        path = get_absolute_path("./test_sin_traj.pth")
        traj: Trajectory = Trajectory[SinusState].load_points_from_file(path, SinusState)
        assert len(traj) == 101  # 101 points in the file
        # should be normalized
        assert traj.get_points()[0].get_time() == 0.0
        assert traj.get_points()[-1].get_time() == 1.0

        path = get_absolute_path("./test_joint_configuration_traj.pth")
        traj: Trajectory = Trajectory[JointConfiguration].load_points_from_file(path, JointConfiguration)
        assert len(traj) == 65  # 65 points in the file

        # should be normalized
        assert traj.get_points()[0].get_time() == 0.0
        assert traj.get_points()[-1].get_time() == 1.0

        # Test exceptional usage (wrong file path)
        with pytest.raises(ValueError):
            Trajectory.load_points_from_file("not_a_file", SinusState)
            Trajectory.load_points_from_file(get_absolute_path("./test_sin_traj.pth"), JointConfiguration)
            Trajectory.load_points_from_file(get_absolute_path("./test_joint_configuration_traj.pth"), SinusState)
