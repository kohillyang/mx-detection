import mxnet as mx


class FrozenBatchNorm2d(mx.gluon.nn.HybridBlock):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, *args, **kwargs):
        super(FrozenBatchNorm2d, self).__init__()
        with self.name_scope():
            self.gamma = self.params.get('gamma', shape=[0, ], init=mx.init.One(), allow_deferred_init=True, grad_req='null')
            self.beta = self.params.get('beta', shape=[0, ], init=mx.init.Zero(), allow_deferred_init=True, grad_req='null')

            self.running_mean = self.params.get('running_mean', shape=[0, ], init=mx.init.Zero(), allow_deferred_init=True, grad_req='null')
            self.running_var = self.params.get('running_var', shape=[0, ], init=mx.init.One(), allow_deferred_init=True, grad_req='null')

        self.eps = 1e-5

    def hybrid_forward(self, F,  x, gamma, beta, running_mean, running_var):
        scale = gamma / (running_var.sqrt() + self.eps)
        bias = beta - running_mean * scale
        scale = scale.reshape((1, -1, 1, 1))
        bias = bias.reshape((1, -1, 1, 1))
        return F.broadcast_add(F.broadcast_mul(x, scale), bias)