# SYSTEM IMPORTS
from typing import List
import numpy as np


# PYTHON PROJECT IMPORTS
from ..module import Module
from ..parameter import Parameter
from .rnn_cells.rnn_cell import RNNCell


class RNN(Module):
    def __init__(self,
                 cell: RNNCell,
                 seq_dim: int = 1,
                 return_sequences: bool = False,
                 backprop_through_time_limit: float = None) -> None:
        self.cell: RNNCell = cell
        self.seq_dim: int = seq_dim
        self.return_sequences: bool = return_sequences

        self.backprop_through_time_limit: float = np.inf
        if backprop_through_time_limit is not None:
            self.backprop_through_time_limit = backprop_through_time_limit

    def init_states(self, batch_size: int) -> np.ndarray:
        return self.cell.init_states(batch_size)

    # X has shape [num_examples, in_dim]
    def forward(self, X: np.ndarray) -> np.ndarray:
        batch_size: int = X.shape[0]
        seq_length: int = X.shape[self.seq_dim]

        states: np.ndarray = self.init_states(batch_size)
        Y_hats_cache: List[np.ndarray] = list()

        for t in range(seq_length):
            states, Y_hat = self.cell.forward(states, X[:, t])
            Y_hats_cache.append(Y_hat)

        if self.return_sequences:
            return np.stack(Y_hats_cache, axis=1)
        else:
            return Y_hats_cache[-1]

    def backwards(self, X: np.ndarray, dLoss_dModule: np.ndarray) -> np.ndarray:
        if not self.return_sequences:
            dLoss_dModule = np.expand_dims(dLoss_dModule, axis=1)

        ####### forward pass, need to cache the states ######
        batch_size: int = X.shape[0]
        seq_length: int = X.shape[self.seq_dim]

        states: np.ndarray = self.init_states(batch_size)
        states_cache: List[np.ndarray] = [states]

        for t in range(seq_length):
            states, _ = self.cell.forward(states, X[:, t])
            states_cache.append(states)
        #####################################################

        # RNN is configurable to only produce a single output or the entire sequence
        output_seq_length: int = dLoss_dModule.shape[1]
        dLoss_dX: np.ndarray = np.zeros_like(X)

        # back propagation through time (aka we unroll the RNN)
        for out_t in range(output_seq_length):

            dLoss_dModule_t: np.ndarray = dLoss_dModule[:, out_t]
            dLoss_dStates_t: np.ndarray = None

            # adjusted_t is what we will use to index the input sequence
            adjusted_t = X.shape[self.seq_dim]-1 if not self.return_sequences else out_t
            for input_t in range(max(0, adjusted_t-self.backprop_through_time_limit),
                                 adjusted_t+1)[::-1]:

                # as we work out way back through the unrolled RNN
                # propagating a given dLoss_dModule_t through the cells that produced
                # the corresponding Y_hat_t, we need to keep track of dLoss_dX
                # and dLoss_dStates_t, as the next cell depends on the previous cell's
                # hidden state.
                dLoss_dX_t, dLoss_dStates_t = self.cell.backwards(X[:,input_t,...],
                                                                  states_cache[input_t],
                                                                  dLoss_dModule_t,
                                                                  dLoss_dStates_t)

                # dLoss_dModule_t is only for the last cell in this chain, so we set it
                # to None so that previous cells don't try to use it
                dLoss_dModule_t = None
                dLoss_dX[:,input_t] += dLoss_dX_t
        return dLoss_dX
                                                                  

    def parameters(self) -> List[Parameter]:
        return self.cell.parameters()

