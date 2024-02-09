from timeit import default_timer as timer

import numpy as np

from mlnn import MLNN
import sample_data

data = sample_data.ghw
X = data['X']
Y = data['Y']
X = X.T
Y = Y.reshape(1, -1)

# provide hidden layers only, input and output sizes are inferred
layers = np.array([8, 4], dtype=int)
n_x = X.shape[0]
n_y = Y.shape[0]
layers = np.insert(layers, 0, n_x)
layers = np.append(layers, n_y)

net = MLNN(layers, h=1)

epochs = int(1e4)

start = timer()
net.fit(X, Y, epochs)
elapsed = timer() - start

a = net.Y_hat(X)
err = net.loss(Y, a)
print(f'{err=}\n{round(elapsed, 3)}s elapsed')

# show progress
"""
for i in range(1, ec+1):
    a = net.Y_hat(X)
    e = net.loss(Y, a)
    if i % (ec//2) == 0:
        print(e//2)
        print(i)
        print(Y)
        print(a)
        print(round(e, 2))
        print()
    net.back_propogation(X, Y)
"""

