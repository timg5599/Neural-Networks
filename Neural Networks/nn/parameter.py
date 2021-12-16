# SYSTEM IMPORTS
import numpy as np


# PYTHON PROJECT IMPORTS


class Parameter(object):
    def __init__(self, V: np.ndarray) -> None:
        self.val: np.ndarray = V
        self.grad: np.ndarray = None

    def reset(self) -> "Parameter":
        self.grad = np.zeros_like(self.val)
        return self

    def step(self, G: np.ndarray) -> "Parameter":
        self.val -= G
        return self.reset()

