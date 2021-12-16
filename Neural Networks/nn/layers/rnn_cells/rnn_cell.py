# SYSTEM IMPORTS
from abc import abstractmethod, ABC
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ...module import Module
from ...parameter import Parameter


class RNNCell(Module, ABC):
    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        self.in_dim: int = in_dim
        self.hidden_dim: int = hidden_dim

    @abstractmethod
    def init_states(self, batch_size: int) -> np.ndarray:
        ...

