# simple-mlnn
A simple binary classification MLNN.

## Usage
Example:
```python
import numpy as np
from mlnn import MLNN
import sample_data
# init and format data
X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
Y = [x[0] | x[1] for x in X]
X, Y = np.array(X), np.array(Y)
X = X.T
Y = Y.reshape(1, -1)
# array of number of neurons in each layer
layers = np.array([X.shape[0], Y.shape[0]])
# initialize the network
net = MLNN(layers, h=1)
# fit to the test data
net.fit(X, Y, epochs=500)
a = net.Y_hat(X)
e = net.loss(Y, a)
# print estimation and error
print([round(i, 2) for i in a[0]])
print(e)
```

Sample output:
```
[0.03, 0.99, 0.99, 1.0]
0.0131356870149288
```

An extended example is given in `example.py`.
## Glossary
$X\in\mathbb{R}^{n_x \times m}$

$Y\in\mathbb{R}^{1 \times m}$
## Design
I decided against support for variable layer activations (anything but sigmoid), multilabel classification, and other loss functions because my gradient descent computation depends on the derivatives of the sigmoid and logistic loss functions. 
## Techniques
- Sigmoid activation: $A=\sigmoid (Z)=\frac{1}{1+e^{-Z}}, \frac{dA}{dZ}=A(1-A)$
- Logistic loss: $L(a, y)=-(yloga+(1-y)log(1-a))$
- Backpropogation of this network currently utilizes batch (not mini-batch) gradient descent. The first two pages of the following document contains relevant gradients: [link](https://cs230.stanford.edu/fall2018/section_files/section3_soln.pdf)
