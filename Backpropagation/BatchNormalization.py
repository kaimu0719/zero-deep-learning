import numpy as np

class BatchNormalization:
    """
    Batch Normalization レイヤ
    -------------------------
    - Ioffe & Szegedy, 2015 (http://arxiv.org/abs/1502.03167)
    - 全結合層・Conv どちらでも使える
    - 推論時は移動平均(running_mean, running_var)を用いる
    """

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        """
        Parameters
        ----------
        - gamma : ndarray
            スケール係数γ(学習可能パラメータ)
        - beta : ndarray
            シフト係数β(学習可能パラメータ)
        - momentum : float
            移動平均の更新式 m = α·m + (1−α)·batch_stat  の α に相当
        - running_mean / running_var : ndarray or None
            推論時に使用する移動平均(学習開始時は None → 初回 forward で初期化)
        """
        # 学習可能パラメータ
        self.gamma = gamma # スケール係数
        self.beta = beta # シフト係数
        self.momentum = momentum # 移動平均の減衰率
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元の判断に使用

        # 推論時に使う移動平均(初期値はNone)
        self.running_mean = running_mean # 平均
        self.running_var = running_var # 分散

        # backward で再利用する中間変数
        self.batch_size = None
        self.xc = None # 中心化後 x (x - μ)
        self.std = None # 標準偏差 σ
        self.dgamma = None # γ 勾配
        self.dbeta = None # β 勾配
    
    def forward(self, x, train_flg=True):
        """
        Parameters
        ----------
        - x : ndarray
            入力(N, D)または(N, C, H, W)
        - train_flg : bool
            True : 学習中 → バッチ統計を使い running_mean, running_var を更新
            False : 推論 → running_mean, running_var を使って正規化
        
        Returns
        -------
        - out : ndarray
            正規化・スケールシフト後の出力
            形状は入力と同じ (N, D) または (N, C, H, W)
        """
        self.input_shape = x.shape # 元形状を保存
        if x.ndim != 2: # Conv層の場合(N, C, H, W) → (N, C*H*W)
            N, C, H, W = x.shape
            x = x.reshape(N, -1)
        
        out = self.__forward(x, train_flg) # 内部 forward 実行

        return out.reshape(*self.input_shape) # 出力を元形状に戻す
        
    def __forward(self, x, train_flg):
        """内部 forward(shape = (N, D)前提/戻り値も(N, D))"""
        # 移動平均が未初期化なら 0 で初期化
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
        
        out = self.gamma * xn + self.beta
        return out
    
    def backward(self, dout):
        """
        Parameters
        ----------
        - dout : ndarray
            上流から来る勾配(入力と同形状)
        
        Returns
        -------
        - dx : ndarray
            入力 x に対する勾配(dL/dx)
            形状は入力と同じ (N, D) または (N, C, H, W)
        """
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)

        return dx
        
    def __backward(self, dout):
        """内部 backward(shape = (N, D) 前提/戻り値も (N, D))"""
        # β, γ の勾配
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)

        # x に関する勾配
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx