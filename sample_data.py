import numpy as np
from pandas import read_csv

bit_inputs = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
xor = {
    'X' : bit_inputs,
    'Y' : np.array([
        g ^ h for g, h in bit_inputs
    ])
}
aand = {
    'X' : bit_inputs,
    'Y' : np.array([
        g & h for g, h in bit_inputs
    ])
}
oor = {
    'X' : bit_inputs,
    'Y' : np.array([
        g | h for g, h in bit_inputs
    ])
}



ghw = read_csv("./data/500_Person_Gender_Height_Weight_Index.csv")
# best performance when only ~10 samples used
X = ghw.loc[:, ["Height", "Weight"]]
# normalization
X = X-X.mean()
Y = ghw.loc[:, "Gender"].apply(lambda l : 0 if l == "Male" else 1)
X = np.array(X)
Y = np.array(Y)
ghw = {
    'X': X,
    'Y' : Y
}




