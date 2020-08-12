import mobula
mobula.config.NVCC = "/usr/local/cuda-10.0/bin/nvcc"
import os
mobula.op.load('FocalLoss', os.path.join(os.path.dirname(__file__), "../"))
import mxnet as mx
import mxnet.autograd as ag
def BCEFocalLoss(x, target, alpha = .25, gamma=2):
    p = x.sigmoid()
    loss = alpha * target * ((1-p)**gamma) * mx.nd.log(p + 1e-11)
    loss = loss + (1-alpha) * (1-target) * (p **gamma) * mx.nd.log(1 - p + 1e-11)
    return -loss

ctx = mx.gpu()
x = mx.nd.random.randn(300, 300, dtype="float64", ctx=ctx)
y = mx.nd.random.randn(300, 300, dtype="float64", ctx=ctx)
x1 = x.copy()
y1 = y.copy()

x.attach_grad()
x1.attach_grad()

with ag.record():
    fl = BCEFocalLoss(x, y, alpha=.25, gamma=2)
    fl_moubula = mobula.op.FocalLoss(alpha=.25, gamma=2, logits=x1, targets=y1)
    fl.backward()
    fl_moubula.backward()
print((fl_moubula - fl).sum())
print(x.grad[0, 0], x1.grad[0, 0])
grad_delta = (x.grad- x1.grad).abs()
print(grad_delta.max())

print(fl[0, 0])
print(fl_moubula[0, 0])
