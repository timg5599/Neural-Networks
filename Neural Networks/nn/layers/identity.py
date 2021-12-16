# SYSTEM IMPORTS
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..parameter import Parameter


class Idendity(Module):

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X

    # no learnable parameters, only need to do one thing:
    #   1) compute (and return) dLoss_dX
    # I did this one for you (derivative is 1)
    def backwards(self, X: np.ndarray, dLoss_dModule: np.ndarray) -> np.ndarray:
        return dLoss_dModule

    def parameters(self) -> List[Parameter]:
        return list()

