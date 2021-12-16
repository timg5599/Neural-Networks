# SYSTEM IMPORTS
from typing import List
from abc import abstractmethod, ABC
import numpy as np


# PYTHON PROJECT IMPORTS
from .parameter import Parameter


class Optimizer(ABC):
    def __init__(self, parameters: List[Parameter], lr: float) -> None:
        self.parameters: List[Parameter] = parameters
        self.lr: float = lr

    def reset(self) -> None:
        for p in self.parameters:
            p.reset()

    @abstractmethod
    def step(self) -> None:
        ...

