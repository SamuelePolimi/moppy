import numpy as np

class DataPoint():
    lr: float
    beta: float
    ld: int
    af: str
    validation: list[float]

    def __init__(self, lr, beta, ld, af, validation) -> None:
        self.lr = lr
        self.beta = beta
        self.validation = validation
        self.ld = ld
        self.af = af

    def get_mean(self):
        return np.mean(self.validation)