import numpy as np

class MLNN:
    def __init__(self, n, h=0.1, W=None, b=None, padding=1e-10):
        # n: 1 x L+1 (L layers) array of number of neurons in each layer
        # h: learning rate
        # W: 1 x L (populated by n_n x n_p arrays)
        # b: 1 x L (populated by n_n x 1 arrays, broadcast to n_n x m)
        # padding: see Y_hat

        self.n = n
        self.h = h
        self.L = n.size - 1
        self.padding = padding

        # init W and b
        self.W = W if W else self.generate_W(n)
        self.b = b if b else self.generate_b(n)

        # init A (for back-prop and debugging)
        self.A = [0 for i in range(self.L)]

        # init Z
        self.Z = [0 for i in range(self.L)]

    def generate_W(self, n):
        W = [0 for i in range(self.L)]
        for l in range(self.L):
            n_p = n[l]
            n_n = n[l+1]
            # W^[l] \in R^{# units in next layer x # units in previous layer}
            W[l] = np.random.uniform(-1, 1, (n_n, n_p))
        return W
    
    def generate_b(self, n):
        b = [0 for i in range(self.L)]
        for l in range(self.L):
            n_n = n[l+1]
            # b^[l] \n R^{# units in next layer}
            b[l] = np.random.uniform(-1, 1, (n_n, 1)) # n_n x 1, but will be broadcast to n_n x m
        return b

    def activation(self, Z):
        # sigmoid activation
        f = lambda x : 1 / (1 + np.exp(-x))
        r = f(Z)
        return r

    def Y_hat(self, X):
        # forward propogation
        def set_Z(l):
            if len(X.shape) == 1:
                b = self.b[l].reshape(-1)
            else:
                b = self.b[l]
            if l == 0:
                A = X
            else:
                A = self.A[l-1]

            self.Z[l] = np.matmul(self.W[l], A) + b
        for l in range(self.L):
            set_Z(l)
            self.A[l] = self.activation(self.Z[l])
            # values of 0 and 1 cause problems with L and dL
            self.A[l][self.A[l]==0] = self.padding
            self.A[l][self.A[l]==1] = 1 - self.padding

        return self.A[-1]

    def loss(self, Y, Y_hat):
        # logistic loss
        m = Y.shape[1]
        loss_f = lambda y, y_hat: -(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
        return 1/m*sum(loss_f(Y, Y_hat)[0])

    def update_network(self, dW, db, l):
        h = self.h
        db = db.reshape(-1, 1)
        self.W[l] -= dW * h
        self.b[l] -= db * h

    def back_propogation(self, X, Y):
        A = self.A
        W = self.W
        m = np.shape(X)[1] # X \in R^{n_x x m}, m is # samples

        # get activations
        self.Y_hat(X) 

        def get_dW(l):
            if l == -1:
                a = X
            else:
                a = A[l-1]
            a = np.transpose(a)
            r = np.matmul(dZ_l, a)
            return r

        # n=L-1 -> 0 (0-based, otherwise n=L -> 1)
        # first cycle because dA_L is different from rest
        dA_l = 1/m * ((1-Y)/(1-A[-1]) - Y/A[-1])
        dZ_l = dA_l*self.activation(self.Z[-1])*(1-self.activation(self.Z[-1]))
        dW_l = get_dW(self.L-1)
        db_l = np.matmul(dZ_l, np.ones(m))
        self.update_network(dW_l, db_l, self.L-1)
        for l in range(self.L-2, -1, -1):
            dA_l = np.matmul(np.transpose(W[l+1]), dZ_l)
            dZ_l = dA_l*self.activation(self.Z[l])*(1-self.activation(self.Z[l]))
            dW_l = get_dW(l-1)
            db_l = np.matmul(dZ_l, np.ones(m))
            self.update_network(dW_l, db_l, l)

    def fit(self, X, Y, epochs):
        for i in range(epochs):
            self.back_propogation(X, Y)





