import mobula
from mobula.const import req
import os

import numpy as np
import mxnet as mx

@mobula.op.register
class RetinaNetTargetGenerator:
    def __init__(self, stride=16,
                 base_size=(32, 32),
                 negative_iou_threshold=.4,
                 positive_iou_threshold=.5,
                 ratios=(.5, 1, 2),
                 scales=(2**0, 2**(1.0/3), 2**(2.0/3)),
                 bbox_norm_coef=(1, 1, 1, 1)
                 ):
        anchors = [[(s * np.sqrt(r), s * np.sqrt(1/r)) for s in scales] for r in ratios]
        anchors_base_wh = np.array(anchors) * np.array(base_size)[np.newaxis, np.newaxis, :]
        anchors_base_wh = anchors_base_wh.reshape(-1, 2)
        self.anchors_base_wh = anchors_base_wh.astype(np.float32)
        self.negative_iou_threshold = negative_iou_threshold
        self.positive_iou_threshold = positive_iou_threshold
        self.stride = stride
        self.bbox_norm_coef = bbox_norm_coef

    def forward(self, data, loc_prediction, cls_prediction, bboxes):
        if self.req[0] == req.null:
            return
        n_batch0, image_h, image_w, _ = data.shape
        n_batch1, num_anchors_times4, feature_h, feature_w = loc_prediction.shape
        n_batch2, num_anchors_times_num_classes, feature_h, feature_w = cls_prediction.shape
        assert int(np.ceil(image_h / self.stride)) == feature_h
        assert int(np.ceil(image_w / self.stride)) == feature_w

        num_anchors = num_anchors_times4 // 4
        assert num_anchors == len(self.anchors_base_wh)
        num_classes = num_anchors_times_num_classes // num_anchors
        n_batch3, bboxes_size, _ = bboxes.shape
        assert n_batch0 == n_batch1 == n_batch2 == n_batch3
        assert bboxes.shape[2] == 5
        assert self.req[0] == req.write

        self.Y[0][:] = 0
        self.Y[1][:] = 0
        self.Y[2][:] = 0
        self.Y[3][:] = 0

        mobula.func.retinanet_target_gen(threads_ref_number=n_batch0 * feature_h *feature_w,
                                         n_batch=n_batch0,
                                         image_h=image_h,
                                         image_w=image_w,
                                         feature_h=feature_h,
                                         feature_w=feature_w,
                                         num_classes=num_classes,
                                         stride=self.stride,
                                         pointer_bboxes=bboxes,
                                         number_of_bboxes=bboxes.shape[1],
                                         negative_iou_threshold=self.negative_iou_threshold,
                                         positive_iou_threshold=self.positive_iou_threshold,
                                         anchors_base_wh=mx.nd.array(self.anchors_base_wh, ctx=data.context),
                                         bbox_norm_coef=mx.nd.array(self.bbox_norm_coef, ctx=data.context),
                                         # anchors_base_wh_size is No. of the anchors.
                                         anchors_base_wh_size=self.anchors_base_wh.shape[0],
                                         loc_targets_output=self.Y[0],
                                         cls_targets_output=self.Y[1],
                                         regmask_targets_output=self.Y[2],
                                         clsmask_targets_output=self.Y[3])

    def backward(self, y0, y1, y2, y3):
        assert self.grad_req[0] == req.null
        assert self.grad_req[1] == req.null
        assert self.grad_req[2] == req.null
        assert self.grad_req[3] == req.null

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[1], in_shape[2],
                          in_shape[1], in_shape[2]]