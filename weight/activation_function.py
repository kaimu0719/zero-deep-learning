import numpy as np
import matplotlib.pyplot as plt

# シグモイド関数の定義
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU関数の定義
def relu(x):
    return np.maximum(0, x)

# tanh関数の定義
def tanh(x):
    return np.tanh(x)