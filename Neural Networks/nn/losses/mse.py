# SYSTEM IMPORTS
import numpy as np


# PYTHON PROJECT IMPORTS
from ..lossfunc import LossFunction


# This is our first error (or "loss") function. MSE stands for "mean squared error"
# and the value that it returns is:
#
#   f(Y_hat, Y_gt) = mean(||y_hat_i - y_gt_i||_2^2)
#
class MSE(LossFunction):

    def forward(self, Y_hat: np.ndarray, Y_gt: np.ndarray) -> float:
        assert(Y_hat.shape == Y_gt.shape)
        return np.sum((Y_hat - Y_gt)**2) / (2*Y_hat.shape[0])

    def backwards(self, Y_hat: np.ndarray, Y_gt: np.ndarray) -> np.ndarray:
        return (Y_hat - Y_gt) / Y_hat.shape[0]

