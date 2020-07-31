import mxnet as mx

from ._resnetv1b import resnet101_v1b, resnet50_v1b, resnet152_v1b


class PyramidNeck(mx.gluon.nn.HybridBlock):
    def __init__(self, feature_dim=256):
        super(PyramidNeck, self).__init__()
        self.fpn_p5_1x1 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=1, prefix="fpn_p5_1x1_")
        self.fpn_p4_1x1 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=1, prefix="fpn_p4_1x1_")
        self.fpn_p3_1x1 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=1, prefix="fpn_p3_1x1_")
        self.fpn_p2_1x1 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=1, prefix="fpn_p2_1x1_")

        self.fpn_p6 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=3, padding=1, strides=2, prefix="fpn_p6_")
        self.fpn_p5 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=3, padding=1, prefix="fpn_p5_")
        self.fpn_p4 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=3, padding=1, prefix="fpn_p4_")
        self.fpn_p3 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=3, padding=1, prefix="fpn_p3_")
        self.fpn_p2 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=3, padding=1, prefix="fpn_p2_")

    def hybrid_forward(self, F, res2, res3, res4, res5):
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


class FPNResNetV1(mx.gluon.nn.HybridBlock):
    def __init__(self, feature_dim=256, num_layers=50, sync_bn=False, num_devices=None, pretrained=True):
        super(FPNResNetV1, self).__init__(prefix="resnetv1")
        self.eps = 1e-5
        feat_kwargs = {}
        if sync_bn is True:
            assert num_devices is not None, "num_devices is not given while sync_bn is set."
            assert isinstance(num_devices, int)
            feat_kwargs["norm_layer"] = mx.gluon.contrib.nn.SyncBatchNorm
            feat_kwargs["norm_kwargs"] = {"num_devices": num_devices}
        assert num_layers in (50, 101, 152)
        if num_layers == 50:
            feat = resnet50_v1b(pretrained=pretrained, use_global_stats=False, **feat_kwargs)
        elif num_layers == 101:
            feat = resnet101_v1b(pretrained=pretrained, use_global_stats=False, **feat_kwargs)
        elif num_layers == 152:
            feat = resnet152_v1b(pretrained=pretrained, use_global_stats=False, **feat_kwargs)
        else:
            raise ValueError("num_layers is not supported, you can implement it by yourselves.")
        self.feat = feat
        self.mean = self.params.get('mean', shape=[1, 3, 1, 1],
                                    init=mx.init.Zero(),
                                    allow_deferred_init=False, grad_req='null')
        self.std = self.params.get('std', shape=[1, 3, 1, 1],
                                   init=mx.init.One(),  # mx.nd.array(),
                                   allow_deferred_init=False, grad_req='null')
        self.mean._load_init(mx.nd.array([[[[0.485]], [[0.456]], [[0.406]]]]), ctx=mx.cpu())
        self.std._load_init(mx.nd.array([[[[0.229]], [[0.224]], [[0.225]]]]), ctx=mx.cpu())
        self.neck = PyramidNeck(feature_dim=feature_dim)

    def hybrid_forward(self, F, x, mean, std=None):
        input = F.transpose(x, (0, 3, 1, 2))
        x = input / 255.0
        x = F.broadcast_sub(x, mean)
        x = F.broadcast_div(x, std)
        x = self.feat.conv1(x)
        x = self.feat.bn1(x)
        x = self.feat.relu(x)
        x = self.feat.maxpool(x)

        res2 = self.feat.layer1(x)
        res3 = self.feat.layer2(res2)
        res4 = self.feat.layer3(res3)
        res5 = self.feat.layer4(res4)

        return self.neck(res2, res3, res4, res5)


