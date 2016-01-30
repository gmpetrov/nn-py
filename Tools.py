import math

class Tools:
    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(x * -1))

    def sigmoidDerivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self, x):
        return math.tanh(x)

    def tanhDerivative(self, x):
        return 1.0 - self.tanh(x) * self.tanh(x)
        

