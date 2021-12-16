# SYSTEM IMPORTS
from typing import List, Tuple
import numpy as np


# PYTHON PROJECT IMPORTS
from ...module import Module
from ...parameter import Parameter
from .rnn_cell import RNNCell
from ..sigmoid import Sigmoid
from ..tanh import Tanh


class VanillaRNNCell(RNNCell):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 hidden_activation: Module = None,
                 output_activation: Module = None) -> None:
        super().__init__(in_dim, hidden_dim)
        self.out_dim: int = out_dim
        self.hidden_activation: Module = hidden_activation
        if self.hidden_activation is None:
            self.hidden_activation = Tanh()     # convention is Tanh
        self.output_activation: Module = output_activation
        if self.output_activation is None:
            self.output_activation = Sigmoid()  # convention is Sigmoid

        self.W: Parameter = Parameter(np.random.randn(self.hidden_dim, self.hidden_dim))
        self.U: Parameter = Parameter(np.random.randn(self.in_dim, self.hidden_dim))
        self.V: Parameter = Parameter(np.random.randn(self.hidden_dim, self.out_dim))

        self.b_state: Parameter = Parameter(np.random.randn(1, self.hidden_dim))
        self.b_out: Parameter = Parameter(np.random.randn(1, self.out_dim))

    # the initial state is a bunch of zeros (convention)
    # we want to allocated a separate initial state for each sequence in the batch
    def init_states(self, batch_size: int) -> np.ndarray:
        return np.zeros((batch_size, self.hidden_dim), dtype=float)

    # this predicts the next timestep for a batch of sequences
    def forward(self, H_t_minus_1: np.ndarray,
                X_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Z_state: np.ndarray = X_t.dot(self.U.val) + H_t_minus_1.dot(self.W.val) + self.b_state.val
        H_t: np.ndarray = self.hidden_activation.forward(Z_state)
        Z_out: np.ndarray = H_t.dot(self.V.val) + self.b_out.val
        Y_hat: np.ndarray = self.output_activation.forward(Z_out)

        return H_t, Y_hat

    def backwards(self, X_t: np.ndarray, H_t_minus_1: np.ndarray,
                  dLoss_dModule_t: np.ndarray,
                  dLoss_dStates_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        # TODO: I really tried so many things but couldn't get it to work...

        dL_dHt: np.ndarray = np.zeros_like(H_t_minus_1)
        if dLoss_dModule_t is not None:
            pass
        if dLoss_dStates_t is not None:
            pass
        dLoss_dH_t_minus_1: np.ndarray = np.zeros_like(H_t_minus_1)
        dLoss_dX_t: np.ndarray = np.zeros_like(X_t)
        dLoss_dH_t_minus_1 = np.dot(dL_dHt,self.W.val.T)
        dLoss_dX_t= np.dot(dL_dHt, self.U.val.T)
        dLoss_dV_t = np.dot(dL_dHt, self.V.val.T)

        self.W.grad= np.dot(H_t_minus_1.T,dL_dHt)
        self.b.grad = np.sum(dL_dHt,axis=0)


        return dLoss_dX_t, dLoss_dH_t_minus_1

    def parameters(self) -> List[Parameter]:
        return [self.W, self.U, self.V, self.b_state, self.b_out]

