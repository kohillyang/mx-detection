import gluoncv
import mxnet as mx
from models.backbones.hrnet._hrnet import HighResolutionClsNet, get_hrnet


class HRNetW32(mx.gluon.nn.HybridBlock):
    def __init__(self, pretrained=False, norm_layer=mx.gluon.nn.BatchNorm, norm_kwargs=None, neck=None):
        super(HRNetW32, self).__init__()
        self.base_net = get_hrnet(model_name="w32", pretrained=pretrained,
                                  norm_layer=norm_layer,
                                  norm_kwargs=norm_kwargs)
        self.mean = self.params.get('mean', shape=[1, 3, 1, 1],
                                    init=mx.init.Zero(),
                                    allow_deferred_init=False, grad_req='null')
        self.std = self.params.get('std', shape=[1, 3, 1, 1],
                                   init=mx.init.One(),  # mx.nd.array(),
                                   allow_deferred_init=False, grad_req='null')
        self.mean._load_init(mx.nd.array([[[[0.485]], [[0.456]], [[0.406]]]]), ctx=mx.cpu())
        self.std._load_init(mx.nd.array([[[[0.229]], [[0.224]], [[0.225]]]]), ctx=mx.cpu())
        self.neck = neck

    def hybrid_forward(self, F, x, mean, std):
        input = F.transpose(x, (0, 3, 1, 2))
        x = input / 255.0
        x = F.broadcast_sub(x, mean)
        x = F.broadcast_div(x, std)
        y_list = super(HighResolutionClsNet, self.base_net).hybrid_forward(F, x)
        if self.neck is not None:
            return self.neck(*y_list)
        else:
            return y_list


if __name__ == '__main__':
    net = HRNetW32()
    net.initialize()
    y=net(mx.nd.zeros(shape=(1, 512, 512, 3)))
    for x in y:
        print(x.shape)
    sym = net(mx.sym.var(name="data"))
    mx.viz.plot_network(sym[0]).view()