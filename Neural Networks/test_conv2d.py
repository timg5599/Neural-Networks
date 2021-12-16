from tqdm import tqdm
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

from nn.module import Module
from nn.lossfunc import LossFunction
from nn.layers.conv2d import Conv2d
from nn.layers.dense import Dense
from nn.layers.flatten import Flatten
from nn.layers.maxpool2d import MaxPool2d
from nn.layers.relu import ReLU
from nn.layers.softmax import Softmax
from nn.models.sequential import Sequential
from nn.optimizers.sgd import SGD
from nn.losses.cross_entropy import CategoricalCrossEntropy


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

num_examples: int =  200
num_classes: int = 3

img_height: int = 10
img_width: int = 10
num_channels: int = 3

kernel_size: int = 3
num_kernels: int = 4

pool_size: Tuple[int, int] = (2, 2)

lr: float = 0.1
max_epochs: int = 100

X: np.ndarray = np.random.randn(num_examples, img_height, img_width, num_channels)

classes: np.ndarray = np.random.randint(num_classes, size=num_examples)
Y_gt: np.ndarray = np.zeros((num_examples, num_classes), dtype=float)
Y_gt[np.arange(num_examples), classes] = 1


m: Sequential = Sequential()
m.add(Conv2d(num_kernels=num_kernels,
             num_channels=num_channels,
             kernel_size=kernel_size))
m.add(MaxPool2d(pool_size=pool_size))
m.add(ReLU())
m.add(Flatten())
m.add(Dense(64, num_classes))
m.add(Softmax())

optim: SGD = SGD(m.parameters(), lr)
loss_func: LossFunction = CategoricalCrossEntropy()

losses = list()
for i in tqdm(list(range(max_epochs)), desc="checking gradients"):

    optim.reset()
    Y_hat = m.forward(X)
    losses.append(loss_func.forward(Y_hat, Y_gt))
    m.backwards(X, loss_func.backwards(Y_hat, Y_gt))

    grad_check(X, Y_gt, m, loss_func, epsilon=1e-6)
    optim.step()


plt.plot(losses)
plt.show()

msg = "INFO: If you see this message (AND you have grad_check enabled " +\
      " AND if MAX_EPOCHS > 0 then your code is WORKING"
print(msg)

