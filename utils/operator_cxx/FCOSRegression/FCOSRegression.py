import mobula
from mobula.const import req
import os

import numpy as np


@mobula.op.register
class FCOSRegression:
    def __init__(self, stride):
        self.stride = stride

    def forward(self, loc_pred, cls_pred):
        if self.req[0] == req.null:
            return
        nbatch, number_of_classes_no_background, feature_h, feature_w = cls_pred.shape
        if self.req[0] == req.add:
            assert False
        else:
            self.y[:] = 0
            mobula.func.fcos_target_regression(outsize=self.y.size,
                                               pointer_loc_pred=loc_pred,
                                               pointer_cls_pred=cls_pred,
                                               nbatch=nbatch,
                                               number_of_classes_no_background=number_of_classes_no_background,
                                               feature_h=feature_h,
                                               feature_w=feature_w,
                                               stride=self.stride,
                                               output=self.y)

    def backward(self, rois):
        pass    # nothing need to do

    def infer_shape(self, in_shape):
        nbatch, number_of_classes_no_background, feature_h, feature_w = in_shape[1]
        return in_shape, [(nbatch, number_of_classes_no_background, feature_h, feature_w, 6)]