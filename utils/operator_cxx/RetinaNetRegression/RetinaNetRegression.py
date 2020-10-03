import mobula
from mobula.const import req
import os
import mxnet as mx
import numpy as np


@mobula.op.register
class RetinaNetRegression:
    def __init__(self, base_size=(32, 32),
                 stride=16,
                 ratios=(.5, 1, 2),
                 scales=(2**0, 2**(1.0/3), 2**(2.0/3)),
                 bbox_norm_coef=(1, 1, 1, 1)):
        anchors = [[(s * np.sqrt(r), s * np.sqrt(1/r)) for s in scales] for r in ratios]
        anchors_base_wh = np.array(anchors) * np.array(base_size)[np.newaxis, np.newaxis, :]
        anchors_base_wh = anchors_base_wh.reshape(-1, 2)
        self.anchors_base_wh = anchors_base_wh.astype(np.float32)
        self.stride = stride
        self.bbox_norm_coef = bbox_norm_coef

    def forward(self, data, reg_preds, cls_preds):
        if self.req[0] == req.null:
            return
        nbatch, image_h, image_w, image_c = data.shape
        nbatch, number_of_anchors_times_4, feature_h, feature_w = reg_preds.shape
        nbatch, number_of_classes_times_number_of_anchors, feature_h, feature_w = cls_preds.shape
        assert self.req[0] == req.write
        assert int(np.ceil(image_h / self.stride)) == feature_h
        assert int(np.ceil(image_w / self.stride)) == feature_w
        number_of_anchors = number_of_anchors_times_4 // 4
        number_of_classes = number_of_classes_times_number_of_anchors // number_of_anchors
        assert number_of_anchors == len(self.anchors_base_wh)
        self.y[:] = 0
        mobula.func.retinanet_regression(size=nbatch*feature_h*feature_w, image_h=image_h, image_w=image_w,
                                         n_batch=nbatch, feature_h=feature_h, feature_w=feature_w,
                                         n_anchor=number_of_anchors,
                                         number_of_classes=number_of_classes,
                                         pointer_reg_preds=reg_preds,
                                         pointer_cls_preds=cls_preds,
                                         pointer_bbox_norm_coef=mx.nd.array(self.bbox_norm_coef, ctx=data.context),
                                         stride=self.stride,
                                         anchors_base_wh=mx.nd.array(self.anchors_base_wh, ctx=data.context),
                                         anchors_base_wh_size=self.anchors_base_wh.shape[0],
                                         output=self.y)

    def backward(self, y):
        pass    # nothing need to do

    def infer_shape(self, in_shape):
        nbatch, number_of_anchors_times_4, feature_h, feature_w = in_shape[1]
        nbatch, number_of_classes_times_number_of_anchors, feature_h, feature_w = in_shape[2]
        number_of_anchors = number_of_anchors_times_4 // 4
        number_of_classes = number_of_classes_times_number_of_anchors // number_of_anchors
        return in_shape, [(nbatch, number_of_anchors * number_of_classes, feature_h, feature_w, 6)]