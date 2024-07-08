# THIS IS JUST A TEST CLASS WHAT IT WOULD TAKE TO MAKE A CUSTOM DATASET CLASS FOR A TRAINER

import glob
import torch
from torch.utils.data import Dataset

from moppy.trajectory.state.joint_configuration import JointConfiguration
from moppy.trajectory.trajectory import Trajectory


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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        traj_path = self.data[idx]
        traj = Trajectory.load_points_from_file(traj_path, JointConfiguration)
        ret = torch.zeros(self.traj_len, len(traj.get_points()[0].to_vector()))
        for i, point in enumerate(traj.get_points()):
            ret[i] = point.to_vector()

        return ret, len(traj.get_points())
