# SYSTEM IMPORTS
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..parameter import Parameter


class Sequential(Module):
    def __init__(self, layers: List[Module] = None) -> None:
        self.layers: List[Module] = layers
        if layers is None:
            self.layers = list()

    def add(self, m: Module) -> "Sequential":
        self.layers.append(m)
        return self

    def parameters(self) -> List[Parameter]:
        params: List[Parameters] = list()
        for m in self.layers:
            params.extend(m.parameters())
        return params

    def forward(self, X: np.ndarray) -> np.ndarray:
        for m in self.layers:
            X = m.forward(X)
        return X

    def backwards(self, X: np.ndarray, dLoss_dModule: np.ndarray) -> np.ndarray:
        Xs: List[np.ndarray] = [X]
        for m in self.layers:
            X = m.forward(X)
            Xs.append(X)

        for i,m in enumerate(self.layers[::-1]):
            dLoss_dModule = m.backwards(Xs[-i-2], dLoss_dModule)
        return dLoss_dModule

