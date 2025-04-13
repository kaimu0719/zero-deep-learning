import numpy as np

class AdaGrad:
    """
    AdaGrad法 : 過去に計算された勾配の二乗和を用いて、各パラメータの学習率を調整する手法。
    """
    def __init__(self, lr=0.01):
        """
        AdaGradの初期化メソッド。

        Parameters
        ----------
        - lr : 学習率（learning rate）
        """
        self.lr = lr
        self.h = None # 勾配の二乗和を保存する変数

    def update(self, params, grads):
        """
        パラメータの更新 : パラメータを勾配に基づいて更新する。

        Parameters
        ----------
        - params : パラメータ（辞書形式）
        - grads : 勾配（辞書形式）
        """
        # 初回の更新時にhを初期化
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        # AdaGrad法による更新
        for key in params.keys():
            self.h[key] += grads[key] * grads[key] # 勾配の二乗和を更新
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)