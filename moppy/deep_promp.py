from moppy.promp import ProMP
from trajectory import Trajectory


class DeepProMP(ProMP):

    def __init__(self, name):
        self.name = name

    def get_traj_distribution_at(time: float, trajectory: Trajectory):
        pass

    def encode():
        pass

    def decode():
        pass

    def train():
        pass

    def test():
        pass

    def validate():
        pass

    def __str__(self):
        return f"MP: {self.name}"
