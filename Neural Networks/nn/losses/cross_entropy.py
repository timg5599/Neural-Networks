# SYSTEM IMPORTS
import numpy as np

# PYTHON PROJECT IMPORTS
from ..lossfunc import LossFunction


class CategoricalCrossEntropy(LossFunction):

    # this computes the expected categorical cross entropy loss
    # which is a scalar. I implemented this for you because it required
    # fancy indexing (i.e. Y_hat[np.arange(...), np.argmax(...)])

    # remember, categorical cross entropy is defined for an arbitrary number
    # of classes, but requires that Y_gt is one-hot. This one-hot version
    # can be used to perform an optimization (which I have done)
    # for computing the actual value.
    def forward(self, Y_hat: np.ndarray, Y_gt: np.ndarray) -> float:
        assert (Y_hat.shape == Y_gt.shape)
        return -np.log(Y_hat[np.arange(Y_hat.shape[0]), np.argmax(Y_gt, axis=-1)]).mean()

    # TODO
    # take the derivative of categorical cross entropy with respect to
    # Y_hat. You will need to implement this derivative and then return it.
    # HINT: you can use the knowledge that the ground truth is one-hot
    #       to speed up this computation!
    def backwards(self, Y_hat: np.ndarray, Y_gt: np.ndarray) -> np.ndarray:
        assert (Y_hat.shape == Y_gt.shape)
        return -Y_gt / (Y_hat **10)
