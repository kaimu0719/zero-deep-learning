import numpy as np
import matplotlib.pyplot as plt

# 標準正規分布に従う乱数を生成
W = np.random.randn(10000)

# 前の層のノード数を指定
n = 100

# スケール（標準偏差の値）を計算
xavier_scale = 1 / np.sqrt(n)
he_scale = np.sqrt(2 / n)
print('xavier', xavier_scale)
print('he', he_scale)

# 作図
plt.hist(W, bins=50) # ヒストグラム
plt.title('Standard Gaussian Distribution', fontsize=20) # タイトル
plt.show()