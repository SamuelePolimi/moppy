# THIS IS JUST A TEST CLASS TO TEXT THE TEST DATASET

from torch.utils.data import DataLoader

from deep_promp.TrajectoryDataset import CustomDataset

class Test:
    pass

if __name__ == "__main__":
    dataset = CustomDataset()
    print("Dataset has length: ", len(dataset))
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for traj, len in data_loader:
        print("Trajectory: ", traj)
        print("len: ", len)
