# SYSTEM IMPORTS
from typing import List
from abc import abstractmethod, ABC
import numpy as np


# PYTHON PROJECT IMPORTS
from .parameter import Parameter


class Module(ABC):

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backwards(self, X: np.ndarray, dLoss_dModule: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def parameters(self) -> List[Parameter]:
        ...

