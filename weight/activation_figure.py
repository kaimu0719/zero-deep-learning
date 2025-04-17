import numpy as np
import matplotlib.pyplot as plt

from activation_function import sigmoid, relu, tanh

# 作図用のx軸の点
x_vec = np.arange(-10.0, 10.0, 0.1)

# 作図する
plt.plot(x_vec, tanh(x_vec), label='tanh function')
plt.plot(x_vec, sigmoid(x_vec), label='sigmoid function')
plt.plot(x_vec, relu(x_vec), label='relu function')

# グラフの設定
plt.hlines(0, xmin=-10.0, xmax=10.0, linestyles='--', linewidth=1)
plt.vlines(0, ymin=-1.5, ymax=1.5, linestyles='--', linewidth=1)
plt.ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()