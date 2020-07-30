import mobula
from mobula.const import req
import os

import numpy as np


@mobula.op.register
class FCOSRecall:
    def __init__(self, stride):
        self.stride = stride

    def forward(self, prediction, bboxes):
        if self.req[0] == req.null:
            return
        out = self.y
        nbatch, feature_ch, feature_h, feature_w = prediction.shape
        number_of_bboxes = bboxes.shape[0]
        assert nbatch==1
        if self.req[0] == req.add:
            out_temp = self.F.zeros_like(out)
            mobula.func.fcos_recall(prediction=prediction, feature_n=nbatch,
                                    feature_h=feature_h, feature_w=feature_w, feature_ch=feature_ch,
                                    stride=self.stride,
                                    number_of_bboxes=number_of_bboxes,
                                    gt_bboxes=bboxes,
                                    output=out_temp)
            self.y[:] += out_temp
        else:
            self.y[:] = 0
            mobula.func.fcos_recall(prediction=prediction, feature_n=nbatch,
                                    feature_h=feature_h, feature_w=feature_w, feature_ch=feature_ch,
                                    stride=self.stride,
                                    number_of_bboxes=number_of_bboxes,
                                    gt_bboxes=bboxes,
                                    output=self.y)

    def backward(self, rois):
        pass    # nothing need to do

    def infer_shape(self, in_shape):
        assert len(in_shape[0]) == 4  # nbatch, 6 + number_of_classes, feature_h, feature_w
        nbatch, c, h, w = in_shape[0]
        return in_shape, [(nbatch, 1)]