# SYSTEM IMPORTS
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..parameter import Parameter


class Dense(Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.W: Parameter = Parameter(np.random.randn(in_dim, out_dim))
        self.b: Parameter = Parameter(np.random.randn(1, out_dim))

    # X has shape [num_examples, in_dim]
    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.W.val) + self.b.val

    # because we have learnable parameters here,
    # we need to do 3 things:
    #   1) compute dLoss_dW
    #   2) compute dLoss_db
    #   3) compute (and return) dLoss_dX

    def backwards(self, X: np.ndarray, dLoss_dModule: np.ndarray) -> np.ndarray:
        grad_input = np.dot(dLoss_dModule, self.W.val.T)

        grad_weights = np.dot(X.T, dLoss_dModule)
        grad_biases = dLoss_dModule.sum(axis=0)

        self.W.grad += grad_weights
        self.b.grad += grad_biases

        return grad_input


    def parameters(self) -> List[Parameter]:
        return [self.W, self.b]

