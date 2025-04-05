class AddLayer:
    """
    加算レイヤー : 2つの入力を加算するレイヤー
    """
    def __init__(self):
        # 加算レイヤーには特に初期化が必要なパラメータがないため何もしない
        pass

    def forward(self, x, y):
        """
        順伝播（forward propagation） : 入力された2つの値（xとy）を加算して返す

        Parameters
        ----------
        - x : 1つ目の入力
        - y : 2つ目の入力

        Returns
        -------
        - out : 加算結果
        """
        out = x + y
        return out

    def backward(self, dout):
        """
        逆伝播（backward propagation）: 出力側から伝わってきた勾配（dout）を、それぞれの入力値に伝える。
        (加算の場合、微分係数は1なので、勾配をそのまま入力側に渡す)

        Parameters
        ----------
        - dout : 出力側から伝わってきた勾配

        Returns
        -------
        - dx : 1つ目の入力（x）の勾配
        - dy : 2つ目の入力（y）の勾配 
        """
        dx = dout * 1
        dy = dout * 1

        return dx, dy