import numpy as np


class TrajectoryDataset:
    """
    A trajectory dataset is in essence a numpy array with shape (n, m, k) where n is the number of trajectories, m is the number of time steps and k is the dimension of the state.
    There is a method load which can be used to load the dataset from a file.
    """

    def __init__(self):
        self.trajectories = []

    def load(self, file_path):
        """Load the dataset from a file."""
        dataset = np.load(file_path)
        if len(dataset.shape) != 3:
            raise ValueError("The dataset should have 3 dimensions, but got {}.".format(len(dataset.shape)))

        # create a trajectory object for each trajectory in the dataset
        for trajectory in dataset:
            traj = Trajectory()
            for time_step, state in enumerate(trajectory):
                traj.add_point(time_step, state)
            self.trajectories.append(traj)

    def add_trajectory(self, trajectory):
        """Add a trajectory to the dataset."""
        self.trajectories.append(trajectory)

    def save(self, file_path):
        """Save the dataset to a file."""
        raise NotImplementedError("This method is not implemented yet.")

    def get_dataset(self):
        """Return the dataset as a numpy array."""
        return self.trajectories

class Trajectory:
    """
    A trajectory is a set of tuples (t_i, x_i) where t_i is a time step (0, 1, 2, ...) and x_i is the state at time t_i.
    The state can be a joint configuration or a pose (the end effector position and orientation).
    """

    def __init__(self):
        self.trajectory = []

    def add_point(self, time_step, state):
        """Add a point to the trajectory."""
        self.trajectory.append((time_step, state))

    def get_trajectory(self):
        """Return the trajectory as a list of tuples."""
        return self.trajectory

    def __len__(self):
        return len(self.trajectory)