from typing import Type

import glob
import torch
from torch.utils.data import Dataset

from .state import JointConfiguration, TrajectoryState
from . import Trajectory


class TrajectoryDataset(Dataset):
    def __init__(self,
                 folder_path: str = 'deep_promp/ReachTarget',
                 state_class: Type[TrajectoryState] = JointConfiguration):
        self.imgs_path = folder_path
        self.state_class = state_class
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        for class_path in file_list:
            for traj_path in glob.glob(class_path + "/*.pth"):
                self.data.append(traj_path)
        if len(self.data) == 0:
            raise ValueError("No data found in the folder")
        trajectories = []
        for traj_path in self.data:
            traj = Trajectory.load_points_from_file(traj_path, self.state_class)
            trajectories.append(traj)
        max_len = max([len(traj) for traj in trajectories])
        self.traj_len = max_len

        self.all_trajectories_with_len = []
        for tr in trajectories:
            new_tr = torch.zeros(max_len, len(tr.get_points()[0].to_vector_without_time()) + self.state_class.get_time_dimension())
            for i, point in enumerate(tr):
                new_p = torch.cat((point.to_vector_without_time(), point.get_time()))
                new_tr[i] = new_p
            self.all_trajectories_with_len.append((new_tr, len(tr.get_points())))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.all_trajectories_with_len[idx]
