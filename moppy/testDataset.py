# THIS IS JUST A TEST CLASS TO TEXT THE TEST DATASET

from torch.utils.data import DataLoader

from deep_promp.TrajectoryDataset import CustomDataset

if __name__ == "__main__":
    dataset = CustomDataset()
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for trajectories, lenght in data_loader:
        print("Trajectory len: ", len(trajectories))
        print("lenght: ", lenght)
