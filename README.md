# Usage
An example is given in `example.py`.
# Glossary
$X\in\mathbb{R}^{n_x x m}$
$Y\in\mathbb{R}^{1 x m}$
$h$ denotes the learning rate of the network.
# Design
I decided against support for variable layer activations (anything but sigmoid), multilabel classification, and other loss functions because my gradient descent computation depends on the derivatives of the sigmoid and logistic loss functions. 
# Backpropogation
Backpropogation of this network utilizes batch (not mini-batch) gradient descent. However, stochastic and mini-batch gradient descent could be implemented.
My implementation of gradient descent was based on my class notes but the first two pages of the following document contains the equations necessary: https://cs230.stanford.edu/fall2018/section_files/section3_soln.pdf