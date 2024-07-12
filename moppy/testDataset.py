# THIS IS JUST A TEST CLASS TO TEXT THE TEST DATASET

from torch.utils.data import DataLoader

from moppy.trajectory import TrajectoryDataset

if __name__ == "__main__":
    dataset = TrajectoryDataset()
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for trajectories, lenght in data_loader:
        print("Trajectory len: ", len(trajectories))
        print("lenght: ", lenght)
