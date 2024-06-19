# THIS IS JUST A TEST CLASS WHAT IT WOULD TAKE TO MAKE A CUSTOM DATASET CLASS FOR A TRAINER

import glob
import numpy as np
from torch.utils.data import Dataset

from trajectory.state.joint_configuration import JointConfiguration
from trajectory.trajectory import Trajectory


class CustomDataset(Dataset):
    def __init__(self, folder_path: str = 'deep_promp/ReachTarget'):
        self.imgs_path = folder_path
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        for class_path in file_list:
            for traj_path in glob.glob(class_path + "/*.pth"):
                self.data.append(traj_path)

        trajectories = []
        for traj_path in self.data:
            traj = Trajectory.load_points_from_file(traj_path, JointConfiguration)
            ret = []
            for point in traj.get_points():
                ret.append(point.to_vector())
            trajectories.append(ret)
        max_len = max([len(traj) for traj in trajectories])
        self.traj_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj_path = self.data[idx]
        traj = Trajectory.load_points_from_file(traj_path, JointConfiguration)
        ret = []
        for point in traj.get_points():
            ret.append(point.to_vector())
        traj_len = len(ret)
        while len(ret) < self.traj_len:
            ret.append(np.zeros_like(ret[0]))
        return ret, traj_len

