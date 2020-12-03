from ._mobilenet import get_mobilenet, get_mobilenet_v2
import mxnet as mx


class MobileNetV1(mx.gluon.nn.HybridBlock):
    def __init__(self, neck, **kwargs):
        super(MobileNetV1, self).__init__(prefix="mobilenet_v1_")
        self.feat = get_mobilenet(**kwargs)
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
        y_list = []
        for i in range(len(self.feat.features)):
            x = self.feat.features[i](x)
            y_list.append(x)
        return self.neck(y_list[20], y_list[32], y_list[68], y_list[80])


if __name__ == '__main__':
    _ = MobileNetV1(lambda x:x, multiplier=1.0, pretrained=True)(mx.nd.zeros(shape=(1, 512, 512, 3)))