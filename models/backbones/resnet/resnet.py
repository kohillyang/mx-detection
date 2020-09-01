import mxnet as mx
import gluoncv
from ._resnetv1b import resnet50_v1b, resnet101_v1b, resnet152_v1b


class ResNetV1B(mx.gluon.nn.HybridBlock):
    def __init__(self, neck, num_layers=50, sync_bn=False, num_devices=None,
                 pretrained=True, use_global_stats=True):
        super(ResNetV1B, self).__init__(prefix="resnetv1")
        self.eps = 1e-5
        feat_kwargs = {}
        if sync_bn is True:
            assert num_devices is not None, "num_devices is not given while sync_bn is set."
            assert isinstance(num_devices, int)
            feat_kwargs["norm_layer"] = mx.gluon.contrib.nn.SyncBatchNorm
            feat_kwargs["norm_kwargs"] = {"num_devices": num_devices}
        assert num_layers in (50, 101, 152)
        if num_layers == 50:
            feat = resnet50_v1b(pretrained=pretrained, use_global_stats=use_global_stats, **feat_kwargs)
        elif num_layers == 101:
            feat = resnet101_v1b(pretrained=pretrained, use_global_stats=use_global_stats, **feat_kwargs)
        elif num_layers == 152:
            feat = resnet152_v1b(pretrained=pretrained, use_global_stats=use_global_stats, **feat_kwargs)
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
        self.neck = neck

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