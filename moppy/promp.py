from trajectory import Trajectory


class ProMP:
    """
    MPs are parametric models that
    capture the essential features of a motion while allowing
    for variations and modifications according to different situations.
    """

    def __init__(self, name):
        self.name = name

    def get_traj_distribution_at(time: float, trajectory: Trajectory):
        pass

    def __str__(self):
        return f"MP: {self.name}"
