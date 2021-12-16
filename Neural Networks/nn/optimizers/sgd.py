# SYSTEM IMPORTS
import numpy as np


# PYTHON PROJECT IMPORTS
from ..optimizer import Optimizer
from ..parameter import Parameter


class SGD(Optimizer):

    def step(self) -> None:
        for p in self.parameters:
            p.step(self.lr * p.grad)

