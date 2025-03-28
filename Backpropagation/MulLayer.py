class MulLayer:
    def __init__(self):
        """
        初期化メソッド
        順伝播の際に入力された値を保持するための変数を定義する。
        """
        self.x = None # 順伝播で入力される一つ目の値
        self.y = None # 順伝播で入力される二つ目の値
    
    def forward(self, x, y):
        """
        順伝播（forward propagation）
        2つの入力値を掛け算し、その結果を出力する。

        Parameters:
        - x: 一つ目の入力値
        - y: 二つ目の入力値

        Returns:
        - out: 掛け算の結果
        """
        self.x = x
        self.y = y
        out = x * y

        return out
    
    def backward(self, dout):
        """
        逆伝播（backward propagation）
        出力側から伝わってきた勾配（dout）に対して、入力値それぞれへの勾配を計算して返す。

        Parameters:
        - dout: 出力側からの勾配

        Returns:
        - dx: 入力xに対する勾配（doutに入力値yを掛けたもの）
        - dy: 入力yに対する勾配（doutに入力値xを掛けたもの）
        """
        dx = dout * self.y # xに対する勾配はyを掛ける
        dy = dout * self.x # yに対する勾配はxを掛ける

        return dx, dy

apple = 100
apple_num = 2
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print("合計の値段", price)

# backword
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print("リンゴの値段の微分", dapple)
print("リンゴの個数の微分", dapple_num)
print("消費税の微分", dtax)