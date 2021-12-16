# SYSTEM IMPORTS
import numpy as np


# PYTHON PROJECT IMPORTS
from ..lossfunc import LossFunction


class BCE(LossFunction):

    # TODO: compute the expected binary cross entropy loss!
    def forward(self, Y_hat: np.ndarray, Y_gt: np.ndarray) -> float:
        assert(Y_hat.shape == Y_gt.shape)

        return - (Y_hat * np.log(Y_gt) + (1 - Y_hat) * np.log(1 - Y_gt)).sum().mean()
    # TODO: take the derivative of binary cross entropy with respect to Y_hat
    # you will then need to compute this derivative and pass return it.
    def backwards(self, Y_hat: np.ndarray, Y_gt: np.ndarray) -> np.ndarray:
        assert(Y_hat.shape == Y_gt.shape)
        return -Y_gt/Y_hat + (1 - Y_gt) / (1 - Y_hat)