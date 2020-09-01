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








