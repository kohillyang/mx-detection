import mobula
from mobula.const import req
import os

import numpy as np


@mobula.op.register
class RetinaNetRegressionCUDA:
    def __init__(self, number_of_classes, stride=16,
                 base_size=(32, 32),
                 ratios=(.5, 1, 2),
                 scales=(2**0, 2**(1.0/3), 2**(2.0/3)),
                 topk=200,
                 ):
        anchors = [[(s * np.sqrt(r), s * np.sqrt(1/r)) for s in scales] for r in ratios]
        anchors_base_wh = np.array(anchors) * np.array(base_size)[np.newaxis, np.newaxis, :]
        anchors_base_wh = anchors_base_wh.reshape(-1, 2)
        self.anchors_base_wh = anchors_base_wh.astype(np.float32)
        self.stride = stride
        self.number_of_classes = number_of_classes
        self.topk = topk

    def forward(self, image, cls_prediction, reg_prediction):
        if self.req[0] == req.null:
            return
        out = self.y
        nbatch, image_h, image_w, image_c = image.shape
        nbatch, feature_h, feature_w, number_of_anchors, out_c = feature.shape
        assert nbatch == 1
        if self.req[0] == req.add:
            out_temp = self.F.zeros_like(out)
            mobula.func.retinanet_regression(image_h=image_h, image_w=image_w,
                                             n_batch=nbatch, feature_h=feature_h, feature_w=feature_w,
                                             n_anchor=number_of_anchors, n_ch=out_c, feature=feature,
                                             stride=self.stride,
                                             anchors_base_wh=self.anchors_base_wh,
                                             anchors_base_wh_size=self.anchors_base_wh.shape[0],
                                             cls_threshold=self.cls_threshold,
                                             output=out_temp)
            self.y[:] += out_temp
        else:
            self.y[:] = 0
            mobula.func.retinanet_regression(image_h=image_h, image_w=image_w,
                                             n_batch=nbatch, feature_h=feature_h, feature_w=feature_w,
                                             n_anchor=number_of_anchors, n_ch=out_c, feature=feature,
                                             stride=self.stride,
                                             anchors_base_wh=self.anchors_base_wh,
                                             anchors_base_wh_size=self.anchors_base_wh.shape[0],
                                             cls_threshold=self.cls_threshold,
                                             output=self.y)

    def backward(self):
        pass    # nothing need to do

    def infer_shape(self, in_shape):
        nbatch, feature_h, feature_w, number_of_anchors, out_c = in_shape[1]
        return in_shape, [(nbatch, feature_h * feature_w * (out_c - 4) * number_of_anchors, 6)]