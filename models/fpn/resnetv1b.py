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


class ResNetV1(mx.gluon.nn.HybridBlock):
    def __init__(self, feature_dim=256, num_layers=101, sync_bn=False, num_devices=None, pretrained=True):
        super(ResNetV1, self).__init__(prefix="resnetv1")
        self.eps = 1e-5
        feat_kwargs = {}
        if sync_bn is True:
            assert num_devices is not None, "num_devices is not given while sync_bn is set."
            assert isinstance(num_devices, int)
            feat_kwargs["norm_layer"] = mx.gluon.contrib.nn.SyncBatchNorm
            feat_kwargs["norm_kwargs"] = {"num_devices": num_devices}
        assert num_layers in (50, 101, 152)
        if num_layers == 50:
            feat = resnet50_v1b(pretrained=pretrained, use_global_stats=True, **feat_kwargs)
        elif num_layers == 101:
            feat = resnet101_v1b(pretrained=pretrained, use_global_stats=True, **feat_kwargs)
        elif num_layers == 152:
            feat = resnet152_v1b(pretrained=pretrained, use_global_stats=True, **feat_kwargs)
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
        x = x / 255.0
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


class ASPP(mx.gluon.nn.HybridBlock):

    def __init__(self,  out_channels):
        super().__init__()
        kernel_sizes = [1, 3, 3, 1]
        dilations = [1, 3, 6, 1]
        paddings = [0, 3, 6, 0]
        with self.name_scope():
            self.aspp = mx.gluon.nn.HybridSequential()
            for aspp_idx in range(len(kernel_sizes)):
                conv = mx.gluon.nn.Conv2D(
                    out_channels,
                    kernel_size=kernel_sizes[aspp_idx],
                    strides=1,
                    dilation=dilations[aspp_idx],
                    padding=paddings[aspp_idx])
                self.aspp.add(conv)
            self.gap = mx.gluon.nn.AvgPool2D(1)
            self.aspp_num = len(kernel_sizes)

    def hybrid_forward(self, F, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(self.aspp_num):
            inp = avg_x if (aspp_idx == self.aspp_num - 1) else x
            out.append(F.relu(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].broadcast_like(out[-2])
        out = F.concat(*out, dim=1)
        return out


class IdenticalHybridBlock(mx.gluon.nn.HybridBlock):
    def hybrid_forward(self, F, x, *args, **kwargs):
        return x


class RFPResNetV1(mx.gluon.nn.HybridBlock):
    def __init__(self, feature_dim=256, num_layers=101, sync_bn=False, num_devices=None, pretrained=True):
        super(RFPResNetV1, self).__init__(prefix="resnetv1")
        self.eps = 1e-5
        feat_kwargs = {}
        if sync_bn is True:
            assert num_devices is not None, "num_devices is not given while sync_bn is set."
            assert isinstance(num_devices, int)
            feat_kwargs["norm_layer"] = mx.gluon.contrib.nn.SyncBatchNorm
            feat_kwargs["norm_kwargs"] = {"num_devices": num_devices}
        assert num_layers in (50, 101, 152)
        if num_layers == 50:
            feat = resnet50_v1b(pretrained=pretrained, use_global_stats=True, **feat_kwargs)
        elif num_layers == 101:
            feat = resnet101_v1b(pretrained=pretrained, use_global_stats=True, **feat_kwargs)
        elif num_layers == 152:
            feat = resnet152_v1b(pretrained=pretrained, use_global_stats=True, **feat_kwargs)
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
        self.stage_with_rfp = [False, True, True, True]
        self.aspp = mx.gluon.nn.HybridSequential()

        self.rfp_feat_conv1x1_layer2 = mx.gluon.nn.Conv2D(channels=512, kernel_size=1)
        self.rfp_feat_conv1x1_layer3 = mx.gluon.nn.Conv2D(channels=1024, kernel_size=1)
        self.rfp_feat_conv1x1_layer4 = mx.gluon.nn.Conv2D(channels=2048, kernel_size=1)

        self.aspp_layer2 = ASPP(out_channels=feature_dim//4)
        self.aspp_layer3 = ASPP(out_channels=feature_dim//4)
        self.aspp_layer4 = ASPP(out_channels=feature_dim//4)

        self.rfp_weight_fusion = mx.gluon.nn.Conv2D(channels=1, kernel_size=1, weight_initializer=mx.init.Zero(),
                                                    bias_initializer=mx.init.Zero())

    def hybrid_forward(self, F, x, mean, std=None):
        x = x / 255.0
        x = F.broadcast_sub(x, mean)
        x = F.broadcast_div(x, std)
        x = self.feat.conv1(x)
        x = self.feat.bn1(x)
        x = self.feat.relu(x)
        pooling_output = self.feat.maxpool(x)

        res2 = self.feat.layer1(pooling_output)
        res3 = self.feat.layer2(res2)
        res4 = self.feat.layer3(res3)
        res5 = self.feat.layer4(res4)

        fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6 = self.neck(res2, res3, res4, res5)
        x_without_rfp = [fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6]
        # Assuming that rfp depth is always 2, we just unroll it here.
        fpn_p3_aspp = self.aspp_layer2(fpn_p3)
        fpn_p4_aspp = self.aspp_layer3(fpn_p4)
        fpn_p5_aspp = self.aspp_layer4(fpn_p5)

        rfp_res2 = res2
        rfp_res3 = self.feat.layer2(rfp_res2) + self.rfp_feat_conv1x1_layer2(fpn_p3_aspp)
        rfp_res4 = self.feat.layer3(rfp_res3) + self.rfp_feat_conv1x1_layer3(fpn_p4_aspp)
        rfp_res5 = self.feat.layer4(rfp_res4) + self.rfp_feat_conv1x1_layer4(fpn_p5_aspp)

        x_idx = self.neck(rfp_res2, rfp_res3, rfp_res4, rfp_res5)
        x_new = []
        # Fusion the feature got from reset and after a rfp stage.
        for ft_idx in range(len(x_idx)):
            add_weight = self.rfp_weight_fusion(x_idx[ft_idx]).sigmoid()
            x_new.append(F.broadcast_mul(add_weight, x_idx[ft_idx])
                         + F.broadcast_mul(1 - add_weight, x_without_rfp[ft_idx]))
        x = x_new
        return x

