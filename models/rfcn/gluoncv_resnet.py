# coding: utf-8
# pylint: disable= arguments-differ, too-many-lines
import mxnet as mx
from ._resnet import get_resnet


class ResNetV1(mx.gluon.nn.HybridBlock):
    def __init__(self, pretrained=True):
        super(ResNetV1, self).__init__(prefix="resnetv1")
        self.eps = 1e-5
        feat = get_resnet(version=1, num_layers=50, use_se=False, pretrained=pretrained)
        self.feat = feat
        self.mean = self.params.get('mean', shape=[1, 3, 1, 1],
                                    init=mx.init.Zero(),
                                    allow_deferred_init=False, grad_req='null')
        self.std = self.params.get('std', shape=[1, 3, 1, 1],
                                   init=mx.init.One(),  # mx.nd.array(),
                                   allow_deferred_init=False, grad_req='null')
        self.mean._load_init(mx.nd.array([[[[0.485]], [[0.456]], [[0.406]]]]), ctx=mx.cpu())
        self.std._load_init(mx.nd.array([[[[0.229]], [[0.224]], [[0.225]]]]), ctx=mx.cpu())

    def hybrid_forward(self, F, x, mean, std=None):
        x = x / 255.0
        x = F.broadcast_sub(x, mean)
        x = F.broadcast_div(x, std)
        for i in range(len(self.feat.features) - 1):
            x = self.feat.features[i](x)
        rcnn_feat = self.feat.features[-1](x)
        return x, rcnn_feat
