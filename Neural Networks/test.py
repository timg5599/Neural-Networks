from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from nn.module import Module
from nn.lossfunc import LossFunction
from nn.layers.dense import Dense
from nn.layers.sigmoid import Sigmoid
from nn.models.sequential import Sequential
from nn.optimizers.sgd import SGD
from nn.losses.mse import MSE


np.random.seed(12345)


# numerical gradient checking...this is how we check whether
# our backpropogation works or not. What we do is compute the
# numerical partial derivative with respect to each learnable
# parameter. A numerical derivative can be computed using:
#       df/dx = (f(x+e) - f(x-e))/(2*e)
# if we set "e" to be really small, then we can get a good approx
# of the gradient. We can compare the symbolic gradients
# versus the numerical gradients, and hope they are super close
def grad_check(X: np.ndarray, Y_gt: np.ndarray, m: Module, ef: LossFunction,
               epsilon: float = 1e-4, delta: float = 1e-6) -> None:
    params: List[Parameter] = m.parameters()
    num_grads: List[np.ndarray] = [np.zeros_like(P.val) for P in params]
    sym_grads: List[np.ndarray] = [P.grad for P in params]

    for P, N in zip(params, num_grads):
        for index, v in np.ndenumerate(P.val):
            P.val[index] += epsilon
            N[index] += ef.forward(m.forward(X), Y_gt)

            P.val[index] -= 2*epsilon
            N[index] -= ef.forward(m.forward(X), Y_gt)

            # set param back to normal
            P.val[index] = v
            N[index] /= (2*epsilon)

    ratios: np.ndarray = np.array([np.linalg.norm(SG-NG)/
                                   np.linalg.norm(SG+NG)
                                   for SG, NG in zip(sym_grads, num_grads)], dtype=float)
    if np.sum(ratios > delta) > 0:
        raise RuntimeError("ERROR: failed grad check. delta: [%s], ratios: %s"
            % (delta, ratios))


in_dim: int = 100
out_dim: int = 3
num_examples: int =  200
lr: float = 0.1
max_epochs: int = 10000

X: np.ndarray = np.random.randn(num_examples, in_dim)
Y_gt: np.ndarray = np.random.randn(num_examples, out_dim)


m: Sequential = Sequential()
m.add(Dense(in_dim, out_dim))
m.add(Sigmoid())

optim: SGD = SGD(m.parameters(), lr)
loss_func: MSE = MSE()

losses = list()
for i in tqdm(list(range(max_epochs)), desc="checking gradients"):
    Y_hat = m.forward(X)
    optim.reset()
    losses.append(loss_func.forward(Y_hat, Y_gt))
    m.backwards(X, loss_func.backwards(Y_hat, Y_gt))

    grad_check(X, Y_gt, m, loss_func)
    optim.step()


plt.plot(losses)
plt.show()

msg = "INFO: If you see this message (AND you have grad_check enabled " +\
      " AND if MAX_EPOCHS > 0 then your code is WORKING"
print(msg)

