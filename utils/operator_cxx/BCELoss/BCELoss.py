import mobula
from mobula.const import req
import numpy as np


@mobula.op.register
class BCELoss:
    def forward(self, y, target):
        return (1 + y.exp()).log() - target * y

    def backward(self, dy):
        grad = (self.X[0]).sigmoid() - self.X[1]
        # grad *= 1e-4
        self.dX[0][:] = grad * dy

    def infer_shape(self, in_shape):
        try:
            assert in_shape[0] == in_shape[1]
        except AssertionError as e:
            print(in_shape)
            raise e
        return in_shape, [in_shape[0]]
