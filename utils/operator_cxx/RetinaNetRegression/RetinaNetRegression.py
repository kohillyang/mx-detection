import mobula
from mobula.const import req
import os

import numpy as np


@mobula.op.register
class RetinaNetRegression:
    def __init__(self, number_of_classes, stride=16,
                 base_size=(32, 32),
                 ratios=(.5, 1, 2),
                 scales=(2**0, 2**(1.0/3), 2**(2.0/3)),
                 ):
        anchors = [[(s * np.sqrt(r), s * np.sqrt(1/r)) for s in scales] for r in ratios]
        anchors_base_wh = np.array(anchors) * np.array(base_size)[np.newaxis, np.newaxis, :]
        anchors_base_wh = anchors_base_wh.reshape(-1, 2)
        self.anchors_base_wh = anchors_base_wh.astype(np.float32)
        self.stride = stride
        self.number_of_classes = number_of_classes


    def forward(self, image, reg_preds, cls_preds, topk_indices):
        if self.req[0] == req.null:
            return
        nbatch, image_h, image_w, image_c = image.shape
        nbatch, _, number_of_anchors, feature_h, feature_w = reg_preds.shape
        nbatch, number_of_classes, number_of_anchors, feature_h, feature_w = cls_preds.shape
        nbatch, topk = topk_indices.shape

        assert nbatch == 1
        if self.req[0] == req.add:
            assert False
        else:
            self.y[:] = 0
            mobula.func.retinanet_regression(image_h=image_h,
                                             image_w=image_w,
                                             n_batch=nbatch,
                                             feature_h=feature_h,
                                             feature_w=feature_w,
                                             topk=topk,
                                             number_of_classes=number_of_classes,
                                             pointer_reg_preds=reg_preds,
                                             pointer_cls_preds=cls_preds,
                                             pointer_topk_indices=topk_indices,
                                             stride=self.stride,
                                             anchors_base_wh=self.anchors_base_wh,
                                             anchors_base_wh_size=self.anchors_base_wh.shape[0],
                                             output=self.y)

    def backward(self, y):
        pass    # nothing need to do

    def infer_shape(self, in_shape):
        nbatch, number_of_classes, number_of_anchors, feature_h, feature_w = in_shape[2]
        return in_shape, [(nbatch, feature_h * feature_w * number_of_classes * number_of_anchors, 6)]