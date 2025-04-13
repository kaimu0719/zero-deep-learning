import numpy as np

def smooth_curve(x, beta=2):
    """
    損失関数の平滑化 : 損失関数のグラフを滑らかにするために使用する。

    Parameters
    ----------
    - x : 損失関数の値（Numpy配列）
    - beta : 平滑化の強さ（デフォルトは2）

    Returns
    -------
    - y : 平滑化された損失関数の値（Numpy配列）
    """
    window_len = 11
    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, beta)
    y = np.convolve(w / w.sum(), s, mode='valid')

    return y[5:len(y) - 5]