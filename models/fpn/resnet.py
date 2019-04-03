import mxnet as mx

from ._resnet import get_resnet


class ResNetV1(mx.gluon.nn.HybridBlock):
    def __init__(self, feature_dim=256, pretrained=True):
        super(ResNetV1, self).__init__(prefix="resnetv1")
        self.eps = 1e-5
        feat = get_resnet(version=1, num_layers=101, use_se=False, pretrained=pretrained)
        self.feat = feat
        self.mean = self.params.get('mean', shape=[1, 3, 1, 1],
                                    init=mx.init.Zero(),
                                    allow_deferred_init=False, grad_req='null')
        self.std = self.params.get('std', shape=[1, 3, 1, 1],
                                   init=mx.init.One(),  # mx.nd.array(),
                                   allow_deferred_init=False, grad_req='null')
        self.mean._load_init(mx.nd.array([[[[0.485]], [[0.456]], [[0.406]]]]), ctx=mx.cpu())
        self.std._load_init(mx.nd.array([[[[0.229]], [[0.224]], [[0.225]]]]), ctx=mx.cpu())
        self.fpn_p5_1x1 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=1, prefix="fpn_p5_1x1_")
        self.fpn_p4_1x1 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=1, prefix="fpn_p4_1x1_")
        self.fpn_p3_1x1 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=1, prefix="fpn_p3_1x1_")
        self.fpn_p2_1x1 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=1, prefix="fpn_p2_1x1_")

        self.fpn_p6 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=3, padding=1, strides=2, prefix="fpn_p6_")
        self.fpn_p5 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=3, padding=1, prefix="fpn_p5_")
        self.fpn_p4 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=3, padding=1, prefix="fpn_p4_")
        self.fpn_p3 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=3, padding=1, prefix="fpn_p3_")
        self.fpn_p2 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=3, padding=1, prefix="fpn_p2_")

    def hybrid_forward(self, F, x, mean, std=None):
        x = x / 255.0
        x = F.broadcast_sub(x, mean)
        x = F.broadcast_div(x, std)
        for i in range(len(self.feat.features) - 4):
            x = self.feat.features[i](x)
        features_length = len(self.feat.features)
        res2 = self.feat.features[features_length - 4](x)
        res3 = self.feat.features[features_length - 3](res2)
        res4 = self.feat.features[features_length - 2](res3)
        res5 = self.feat.features[features_length - 1](res4)

        fpn_p5_1x1 = self.fpn_p5_1x1(res5)
        fpn_p4_1x1 = self.fpn_p4_1x1(res4)
        fpn_p3_1x1 = self.fpn_p3_1x1(res3)
        fpn_p2_1x1 = self.fpn_p2_1x1(res2)

        fpn_p5_upsample = F.contrib.BilinearResize2D(fpn_p5_1x1, scale_height=2, scale_width=2)
        fpn_p4_plus = F.ElementWiseSum(*[fpn_p5_upsample, fpn_p4_1x1])
        fpn_p4_upsample = F.contrib.BilinearResize2D(fpn_p4_plus, scale_height=2, scale_width=2)
        fpn_p3_plus = F.ElementWiseSum(*[fpn_p4_upsample, fpn_p3_1x1])
        fpn_p3_upsample = F.contrib.BilinearResize2D(fpn_p3_plus, scale_height=2, scale_width=2)
        fpn_p2_plus = F.ElementWiseSum(*[fpn_p3_upsample, fpn_p2_1x1])
        # FPN feature
        fpn_p6 = self.fpn_p6(res5)
        fpn_p5 = self.fpn_p5(fpn_p5_1x1)
        fpn_p4 = self.fpn_p4(fpn_p4_plus)
        fpn_p3 = self.fpn_p3(fpn_p3_plus)
        fpn_p2 = self.fpn_p2(fpn_p2_plus)

        return fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6
