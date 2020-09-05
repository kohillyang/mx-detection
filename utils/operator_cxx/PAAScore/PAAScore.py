import mobula
from mobula.const import req
import os

import numpy as np


@mobula.op.register
class PAAScore:
    def __init__(self, number_of_classes, stride):
        self.stride = stride
        self.number_of_classes = number_of_classes

    def forward(self, image, reg_preds, cls_preds, anchors_base_wh, gt_boxes, gt_boxes_number):
        if self.req[0] == req.null:
            return
        out = self.y
        nbatch, image_h, image_w, image_c = image.shape
        nbatch, number_of_anchors_times_4, feature_h, feature_w = reg_preds.shape
        nbatch, number_of_anchors_times_number_of_classes, feature_h, feature_w = cls_preds.shape
        number_of_anchors = anchors_base_wh.shape[0]
        number_of_classes = number_of_anchors_times_number_of_classes // number_of_anchors

        if self.req[0] == req.add:
            assert False
        else:
            self.y[:] = 0
            assert gt_boxes.shape[2] == 5
            mobula.func.paa_score(stride=self.stride,
                                  image_h=image_h,
                                  image_w=image_w,
                                  nbatch=nbatch,
                                  feature_h=feature_h,
                                  feature_w=feature_w,
                                  number_of_anchors=number_of_anchors,
                                  number_of_classes=number_of_classes,
                                  gt_boxes_padded_length=gt_boxes.shape[1],
                                  pointer_reg_preds=reg_preds,
                                  pointer_cls_preds=cls_preds,
                                  pointer_gt_boxes=gt_boxes,
                                  pointer_gt_boxes_number=gt_boxes_number,
                                  pointer_anchors_base_wh=anchors_base_wh,
                                  output=self.y)

    def backward(self, y):
        pass    # nothing need to do

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[2]]