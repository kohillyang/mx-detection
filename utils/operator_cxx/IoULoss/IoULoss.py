import mobula
from mobula.const import req
import numpy as np


@mobula.op.register
class IoULoss:
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, preds, targets):
        if self.req[0] == req.null:
            return
        assert preds.shape[self.axis] == 4
        assert targets.shape[self.axis] == 4

        size_before_axis = int(np.prod(preds.shape[:self.axis]))
        size_after_axis = int(np.prod(preds.shape[(self.axis + 1):]))
        if self.req[0] == req.add:
            assert False
        else:
            self.y[:] = 0
            mobula.func.iou_loss_forward(size_before_axis=size_before_axis,
                                         size_after_axis=size_after_axis,
                                         preds=preds, targets=targets, outputs=self.y)

    def backward(self, dy):
        assert self.req[1] == req.null
        preds = self.X[0]
        targets = self.X[1]
        size_before_axis = int(np.prod(preds.shape[:self.axis]))
        size_after_axis = int(np.prod(preds.shape[(self.axis + 1):]))
        if self.req[0] == req.add:
            out_temp = self.F.zeros_like(self.dX[0])
            mobula.func.iou_loss_backward(size_before_axis=size_before_axis, size_after_axis=size_after_axis,
                                          preds=preds, targets=targets, outputs=out_temp)
            self.dX[0] += out_temp
        else:
            self.dX[0][:] = 0
            mobula.func.iou_loss_backward(size_before_axis=size_before_axis, size_after_axis=size_after_axis,
                                          preds=preds, targets=targets, outputs=self.dX[0])

        self.dX[0][:] *= dy

    def infer_shape(self, in_shape):
        size_before_axis = int(np.prod(in_shape[0][:self.axis]))
        size_after_axis = int(np.prod(in_shape[0][(self.axis + 1):]))
        return in_shape, [(size_before_axis, size_after_axis)]
