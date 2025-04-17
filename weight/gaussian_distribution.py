import numpy as np

# 標準偏差を指定
std = 100

# 標準正規分布に従う乱数を生成
W = np.random.randn(10, 100)

print(np.round(np.std(W, axis=1), 4))
print(np.round(np.std(W * std, axis=1), 2))
