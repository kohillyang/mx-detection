import mxnet as mx
from ._resnetv1 import resnet50_v1


class ResNetV1(mx.gluon.nn.HybridBlock):
    def __init__(self, neck, num_layers=50, **kwargs):
        super(ResNetV1, self).__init__(prefix="resnetv1")
        assert num_layers in (50, 101, 152)
        if num_layers == 50:
            feat = resnet50_v1(**kwargs)
        else:
            assert False
        self.feat = feat
        self.mean = self.params.get('mean', shape=[1, 3, 1, 1],
                                    init=mx.init.Zero(),
                                    allow_deferred_init=False, grad_req='null')
        self.std = self.params.get('std', shape=[1, 3, 1, 1],
                                   init=mx.init.One(),  # mx.nd.array(),
                                   allow_deferred_init=False, grad_req='null')
        self.mean._load_init(mx.nd.array([[[[0.485]], [[0.456]], [[0.406]]]]), ctx=mx.cpu())
        self.std._load_init(mx.nd.array([[[[0.229]], [[0.224]], [[0.225]]]]), ctx=mx.cpu())
        self.neck = neck

    def hybrid_forward(self, F, x, mean, std=None):
        input = F.transpose(x, (0, 3, 1, 2))
        x = input / 255.0
        x = F.broadcast_sub(x, mean)
        x = F.broadcast_div(x, std)
        for i in range(4):
            x = self.feat.features[i](x)

        res2 = self.feat.features[4](x)
        res3 = self.feat.features[5](res2)
        res4 = self.feat.features[6](res3)
        res5 = self.feat.features[7](res4)
        return self.neck(res2, res3, res4, res5)

