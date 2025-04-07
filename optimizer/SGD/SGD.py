class SGD:
    """
    確率的勾配降下法（Stochastic Gradient Descent）
    """
    def __init__(self, lr=0.01):
        """
        SGDの初期化メソッド。

        Parameters
        ----------
        - lr : 学習率（learning rate）
        """
        self.lr = lr # 学習率
    
    def update(self, params, grads):
        """
        パラメータの更新 : パラメータを勾配に基づいて更新する。

        Parameters
        ----------
        - params : パラメータ（辞書形式）
        - grads : 勾配（辞書形式）
        """
        for key in params.keys():
            params[key] -= self.lr * grads[key]

