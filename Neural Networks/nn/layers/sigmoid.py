# SYSTEM IMPORTS
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..parameter import Parameter


class Sigmoid(Module):

    # the formula of sigmoid is 1/(1 + e^-x)
    def forward(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    # no learnable parameters, only need to do one thing:
    #   1) compute (and return) dLoss_dX
    def backwards(self, X: np.ndarray, dLoss_dModule: np.ndarray) -> np.ndarray:
        forward = self.forward(X)
        return dLoss_dModule * forward * (1-forward)

    def parameters(self) -> List[Parameter]:
        return list()

