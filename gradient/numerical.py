import numpy as np
from tqdm import tqdm

def numerical_gradient(f, x):
    h = 1e-4 # 微小値
    grad = np.zeros_like(x) # 勾配の初期化

    for idx in tqdm(np.ndindex(x.shape), total=x.size, desc="Calculating gradients"):
        tmp_val = x[idx]

        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)

        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x) # f(x-h)

        # 勾配の計算
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        # 元の値に戻す
        x[idx] = tmp_val

    return grad