# SYSTEM IMPORTS
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..parameter import Parameter


class TimeDistributed(Module):
    def __init__(self, module: Module) -> None:
        self.module: Module = module

    # X has shape [num_examples, in_dim]
    def forward(self, X: np.ndarray) -> np.ndarray:
        batch_dim, sequence_size = X.shape[:2]

        Y_hats: List[np.ndarray] = list()
        for t in range(sequence_size):
            Y_hats.append(self.module.forward(X[:,t]))
        return np.stack(Y_hats, axis=1)

    def backwards(self, X: np.ndarray, dLoss_dModule: np.ndarray) -> np.ndarray:
        batch_dim, sequence_size = dLoss_dModule.shape[:2]
        assert(X.shape[0] == dLoss_dModule.shape[0])
        assert(X.shape[1] == dLoss_dModule.shape[1])

        dLoss_dXs: List[np.ndarray] = list()
        for t in range(sequence_size):
            dLoss_dXs.append(self.module.backwards(X[:,t], dLoss_dModule[:,t]))
        return np.stack(dLoss_dXs, axis=1)

    def parameters(self) -> List[Parameter]:
        return self.module.parameters()

