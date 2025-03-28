class MulLayer:
    def __init__(self):
        """
        順伝播の入力を保持するために使用する
        """
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out
    
    def backward(self, dout):
        dx = dout * self.y # `x`と`y`をひっくり返す
        dy = dout * self.x

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