import numpy as np

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        """
        Adamの初期化メソッド。

        Parameters
        ----------
        - lr : 学習率（learning rate）
        - beta1 : 一次モーメントの指数移動平均の係数（default=0.9）
        - beta2 : 二次モーメントの指数移動平均の係数（default=0.999）
        - iter : 更新回数のカウンタ（default=0）
        - m : 一次モーメントの初期化（default=None）
        - v : 二次モーメントの初期化（default=None）
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
    
    def update(self, params, grads):
        """
        パラメータの更新 : パラメータを勾配に基づいて更新する。

        Parameters
        ----------
        - params : パラメータ（辞書形式）
        - grads : 勾配（辞書形式）
        """
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        # 更新回数をカウント
        self.iter += 1

        # 学習率の調整
        # beta1とbeta2の指数移動平均を考慮した学習率
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            # 一次モーメントと二次モーメントの更新
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            # パラメータの更新
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
