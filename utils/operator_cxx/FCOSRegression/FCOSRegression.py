import mobula
from mobula.const import req
import os

import numpy as np


@mobula.op.register
class FCOSRegression:
    def __init__(self, stride):
        self.stride = stride

    def forward(self, prediction):
        if self.req[0] == req.null:
            return
        out = self.y
        nbatch, feature_ch, feature_h, feature_w = prediction.shape
        assert nbatch==1
        if self.req[0] == req.add:
            out_temp = self.F.zeros_like(out)
            mobula.func.fcos_target_regression(prediction=prediction, feature_n=nbatch,
                                               feature_h=feature_h, feature_w=feature_w, feature_ch=feature_ch,
                                               stride=self.stride,
                                               output=out_temp)
            self.y[:] += out_temp
        else:
            self.y[:] = 0
            mobula.func.fcos_target_regression(prediction=prediction, feature_n=nbatch,
                                               feature_h=feature_h, feature_w=feature_w, feature_ch=feature_ch,
                                               stride=self.stride,
                                               output=self.y)

    def backward(self, rois):
        pass    # nothing need to do

    def infer_shape(self, in_shape):
        assert len(in_shape[0]) == 4  # nbatch, 6 + number_of_classes, feature_h, feature_w
        nbatch, c, h, w = in_shape[0]
        stride = self.stride
        # 4 channel for bbox
        # one channel for centerness
        # No. of classes channels for class id,
        # 6 + 81 channels in total, if coco dataset is used.
        return in_shape, [(nbatch, h * w * (c-5), 6)]