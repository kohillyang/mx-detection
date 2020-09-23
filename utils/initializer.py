from mxnet.initializer import register, Initializer
import mxnet as mx
import numpy as np
@register
class KaMingUniform(Initializer):
    def __init__(self):
        super(KaMingUniform, self).__init__()

    def _init_weight(self, _, arr):
        fan_in = arr.shape[1]
        mx.nd.random.uniform(-1 *np.sqrt(6 / fan_in), np.sqrt(6 / fan_in), shape=arr.shape, out=arr)