import numpy as np
import mxnet as mx
import gluoncv as gl
from mxnet.gluon import nn
import gluoncv

# many are borrowed from https://github.com/ycszen/pytorch-ss/blob/master/gcn.py
from .resnet import resnet50_v1b


class _GlobalConvModule(nn.Block):
    def __init__(self, out_dim, kernel_size):
        super(_GlobalConvModule, self).__init__()
        pad0 = (kernel_size[0] - 1) // 2
        pad1 = (kernel_size[1] - 1) // 2
        # kernel size had better be odd number so as to avoid alignment error
        super(_GlobalConvModule, self).__init__()

        self.conv_l1 = nn.Conv2D(out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2D(out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2D(out_dim, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2D(out_dim, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x = x_l + x_r
        return x


class _BoundaryRefineModule(nn.Block):
    def __init__(self, dim):
        super(_BoundaryRefineModule, self).__init__()
        self.relu = nn.Activation('relu')
        self.conv1 = nn.Conv2D(dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2D(dim, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        out = x + residual
        return out


class GCN(nn.Block):
    def Sequential(self, *args):
        net = nn.HybridSequential()
        with net.name_scope():
            for a in args:
                net.add(a)
        return net

    def __init__(self, resnet_block, num_classes, input_size=(368, 368)):
        super(GCN, self).__init__()

        resnet = resnet_block
        self.layer0 = self.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.layer1 = self.Sequential(resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcm1 = _GlobalConvModule(num_classes, (7, 7))
        self.gcm2 = _GlobalConvModule(num_classes, (7, 7))
        self.gcm3 = _GlobalConvModule(num_classes, (7, 7))
        self.gcm4 = _GlobalConvModule(num_classes, (7, 7))

        self.brm1 = _BoundaryRefineModule(num_classes)
        self.brm2 = _BoundaryRefineModule(num_classes)
        self.brm3 = _BoundaryRefineModule(num_classes)
        self.brm4 = _BoundaryRefineModule(num_classes)
        self.brm5 = _BoundaryRefineModule(num_classes)
        self.brm6 = _BoundaryRefineModule(num_classes)
        self.brm7 = _BoundaryRefineModule(num_classes)
        self.brm8 = _BoundaryRefineModule(num_classes)
        self.brm9 = _BoundaryRefineModule(num_classes)

        #         initialize_weights(self.gcm1, self.gcm2, self.gcm3, self.gcm4, self.brm1, self.brm2, self.brm3,
        #                            self.brm4, self.brm5, self.brm6, self.brm7, self.brm8, self.brm9)

        self.ishape = input_size

        self.mean = self.params.get('mean', shape=[1, 3, 1, 1],
                                    init=mx.init.Zero(),
                                    allow_deferred_init=False, grad_req='null')
        self.std = self.params.get('std', shape=[1, 3, 1, 1],
                                   init=mx.init.One(),  # mx.nd.array(),
                                   allow_deferred_init=False, grad_req='null')
        self.mean._load_init(mx.nd.array([[[[0.485]], [[0.456]], [[0.406]]]]), ctx=mx.cpu())
        self.std._load_init(mx.nd.array([[[[0.229]], [[0.224]], [[0.225]]]]), ctx=mx.cpu())


    def forward(self, x):
        F = mx.nd
        x = F.transpose(x, (0, 3, 1, 2))
        x = x / 255.0
        x = F.broadcast_sub(x, self.mean.data(x.context))
        x = F.broadcast_div(x, self.std.data(x.context))
        H, W = x.shape[2:4]
        # if x: 512
        fm0 = self.layer0(x)  # 256
        fm1 = self.layer1(fm0)  # 128
        fm2 = self.layer2(fm1)  # 64
        fm3 = self.layer3(fm2)  # 32
        fm4 = self.layer4(fm3)  # 16

        gcfm1 = self.brm1(self.gcm1(fm4))  # 16
        gcfm2 = self.brm2(self.gcm2(fm3))  # 32
        gcfm3 = self.brm3(self.gcm3(fm2))  # 64
        gcfm4 = self.brm4(self.gcm4(fm1))  # 128

        #         print(gcfm2.shape,self.upsample(gcfm1, (H//8,W//8),mode='bilinear').shape,x.shape)
        fs1 = self.brm5(self.upsample(gcfm1, (H // 8, W // 8), mode='bilinear') + gcfm2)  # 32
        fs2 = self.brm6(self.upsample(fs1, (H // 8, W // 8), mode='bilinear') + gcfm3)  # 64
        fs3 = self.brm7(self.upsample(fs2, (H // 4, W // 4), mode='bilinear') + gcfm4)  # 128
        fs4 = self.brm8(self.upsample(fs3, (H // 2, W // 2), mode='bilinear'))  # 256
        out = self.brm9(self.upsample(fs4, (H, W), mode='bilinear'))  # 512

        return out

    def upsample(self, input, dsize, mode="bilinear"):
        return mx.nd.contrib.BilinearResize2D(input, height=dsize[0], width=dsize[1])


def DRN50_GCN(num_classes=3):
    from gluoncv.model_zoo import resnet50_v1b as resnet50_v1b_ori
    resnet_block = resnet50_v1b(dilated=True, pretrained=True)
    return GCN(resnet_block=resnet_block, num_classes=num_classes)


if __name__ == "__main__":
    import time

    gpu_id = 8
    net = DRN50_GCN(num_classes=24)
    data = mx.nd.zeros(shape=(1, 3, 512, 512))
    x = mx.nd.zeros(shape=(1, 3, 512, 512), ctx=mx.gpu(gpu_id), dtype="float32", )
    net.initialize(init=mx.init.Normal())
    net.collect_params().reset_ctx(mx.gpu(gpu_id))

    y = net(x)
    net.collect_params().save("output/tmp.params")
    while True:
        t0 = time.time()
        y = net(x)
        mx.nd.waitall()
        print(time.time() - t0)
    print(y.shape)














