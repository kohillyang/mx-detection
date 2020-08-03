import mobula
from mobula.const import req
import os

import numpy as np


@mobula.op.register
class RetinaNetTargetGenerator:
    def __init__(self, number_of_classes, stride=16,
                 base_size=(32, 32),
                 negative_iou_threshold=.4,
                 positive_iou_threshold=.5,
                 ratios=(.5, 1, 2),
                 scales=(2**0, 2**(1.0/3), 2**(2.0/3)),
                 ):
        anchors = [[(s * np.sqrt(r), s * np.sqrt(1/r)) for s in scales] for r in ratios]
        anchors_base_wh = np.array(anchors) * np.array(base_size)[np.newaxis, np.newaxis, :]
        anchors_base_wh = anchors_base_wh.reshape(-1, 2)
        self.anchors_base_wh = anchors_base_wh.astype(np.float32)
        self.negative_iou_threshold = negative_iou_threshold
        self.positive_iou_threshold = positive_iou_threshold
        self.stride = stride
        self.number_of_classes = number_of_classes

    def forward(self, image, bboxes):
        if self.req[0] == req.null:
            return
        out = self.y
        feature_h, feature_w, number_of_anchors, out_c = out.shape
        assert bboxes.shape[1] == 5
        if self.req[0] == req.add:
            out_temp = self.F.zeros_like(out)
            mobula.func.retinanet_target_gen(image_h=image.shape[0], image_w=image.shape[1],
                                             feature_h=feature_h, feature_w=feature_w, feature_ch=out_c,
                                             stride=self.stride, bboxes=bboxes,
                                             number_of_bboxes=bboxes.shape[0],
                                             negative_iou_threshold=self.negative_iou_threshold,
                                             positive_iou_threshold=self.positive_iou_threshold,
                                             anchors_base_wh=self.anchors_base_wh,
                                             anchors_base_wh_size=self.anchors_base_wh.shape[0],
                                             output=out_temp)
            self.y[:] += out_temp
        else:
            self.y[:] = 0
            mobula.func.retinanet_target_gen(image_h=image.shape[0], image_w=image.shape[1],
                                             feature_h=feature_h, feature_w=feature_w, feature_ch=out_c,
                                             stride=self.stride, bboxes=bboxes,
                                             number_of_bboxes=bboxes.shape[0],
                                             negative_iou_threshold=self.negative_iou_threshold,
                                             positive_iou_threshold=self.positive_iou_threshold,
                                             anchors_base_wh=self.anchors_base_wh,
                                             anchors_base_wh_size=self.anchors_base_wh.shape[0],
                                             output=self.y)

    def backward(self):
        pass    # nothing need to do

    def infer_shape(self, in_shape):
        assert len(in_shape[0]) == 3  # Image Height * Image Width * 3
        assert len(in_shape[1]) == 2  # number of bboxes * 5
        h, w, c = in_shape[0]
        stride = self.stride
        # one channel for cls mask.
        # one channel for regression mask.
        # 4 channel for bbox
        # No. of classes channels for class id,
        # 6 + 81 channels in total, if coco dataset is used.
        output_h = int(np.ceil(1.0 * h / stride))
        output_w = int(np.ceil(1.0 * w / stride))
        return in_shape, [(output_h, output_w, self.anchors_base_wh.shape[0], 6 + self.number_of_classes - 1)]