import mobula
from mobula.const import req
import os

import numpy as np


@mobula.op.register
class FCOSTargetGenerator:
    def __init__(self, stride, min_distance, max_distance, number_of_classes):
        self.stride = stride
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.number_of_classes = number_of_classes

    def forward(self, image, bboxes):
        if self.req[0] == req.null:
            return
        out = self.y
        feature_h, feature_w, out_c = out.shape
        assert bboxes.shape[1] == 5
        if self.req[0] == req.add:
            out_temp = self.F.zeros_like(out)
            mobula.func.fcos_target_gen(feature_h=feature_h, feature_w=feature_w, feature_ch=out_c,
                                        stride=self.stride, bboxes=bboxes,
                                        number_of_bboxes=bboxes.shape[0],
                                        distance_min=self.min_distance,
                                        distance_max=self.max_distance,
                                        output=out_temp)
            self.y[:] += out_temp
        else:
            self.y[:] = 0
            mobula.func.fcos_target_gen(feature_h=feature_h, feature_w=feature_w, feature_ch=out_c,
                                        stride=self.stride, bboxes=bboxes,
                                        number_of_bboxes=bboxes.shape[0],
                                        distance_min=self.min_distance,
                                        distance_max=self.max_distance,
                                        output=self.y)

    def backward(self):
        pass    # nothing need to do

    def infer_shape(self, in_shape):
        assert len(in_shape[0]) == 3  # Image Height * Image Width * 3
        assert len(in_shape[1]) == 2  # number of bboxes * 5
        h, w, c = in_shape[0]
        stride = self.stride
        assert h % stride == 0
        assert w % stride == 0
        # one channel for mask
        # 4 channel for bbox
        # one channel for centerness
        # No. of classes channels for class id,
        # 6 + 81 channels in total, if coco dataset is used.
        return in_shape, [(h // stride, w // self.stride, 6 + self.number_of_classes)]