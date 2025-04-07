import numpy as np

class Momentum:
    """
    モーメンタム法 : 勾配降下法の一種で、過去の勾配を考慮してパラメータを更新する手法。
    """
    def __init__(self, lr=0.01, momentum=0.9):
        """
        Momentumの初期化メソッド。

        Parameters
        ----------
        - lr : 学習率（learning rate）
        - momentum : モーメンタム係数（momentum coefficient）
        """
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        """
        パラメータの更新 : パラメータを勾配に基づいて更新する。

        Parameters
        ----------
        - params : パラメータ（辞書形式）
        - grads : 勾配（辞書形式）
        """
        # 初回の更新時にvを初期化
        if self.v is None:
            self.v = {}
            for key in params.keys():
                self.v[key] = np.zeros_like(params[key])
        
        # モーメンタム法による更新
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]