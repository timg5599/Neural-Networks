# SYSTEM IMPORTS
from abc import abstractmethod, ABC
import numpy as np


# PYTHON PROJECT IMPORTS


class LossFunction(ABC):

    @abstractmethod
    def forward(self, Y_hat: np.ndarray, Y_gt: np.ndarray) -> float:
        ...

    @abstractmethod
    def backwards(self, Y_hat: np.ndarray, Y_gt: np.ndarray) -> np.ndarray:
        ...

