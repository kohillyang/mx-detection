from mxnet.initializer import register, Initializer
import mxnet as mx
import numpy as np
import torch
@register
class KaMingUniform(Initializer):
    def __init__(self):
        super(KaMingUniform, self).__init__()

    def _init_weight(self, _, arr):
        w = torch.nn.init.kaiming_uniform_(torch.zeros(size=arr.shape), a=1).numpy()
        arr[:] = mx.nd.array(w)