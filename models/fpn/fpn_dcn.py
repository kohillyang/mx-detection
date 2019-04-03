import mxnet as mx
import mxnet.gluon.nn as nn
from utils.pyramid_proposal import pyramid_proposal
import numpy as np
import mxnet.ndarray as nd


class FPNDeformablePSROIPooling(nn.HybridBlock):
    def __init__(self, stride, pooled_height=7, pooled_width=7, output_dim=256, **kwargs):
        super(FPNDeformablePSROIPooling, self).__init__(**kwargs)
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.stride = stride
        self.output_dim = output_dim

        with self.name_scope():
            # The pooled size is 7 and the dim is 256, then the size is 7*7*256
            # Specifying the in_units here can avoid deferred initialization.
            self.fc = nn.Dense(7 * 7 * 2, in_units=256 * 7 * 7,  prefix="")
            self.fc.weight.lr_mult = 0.01
            self.fc.bias.lr_mult = 0.01

        self.fc.collect_params().setattr('lr_mult', 0.01)

    def hybrid_forward(self, F, rois, feat):
        roi_offset_t = F.contrib.DeformablePSROIPooling(data=feat, rois=rois,
                                                        group_size=1, pooled_size=7,
                                                        sample_per_part=4, no_trans=True,
                                                        part_size=7, output_dim=256,
                                                        spatial_scale=1.0 / self.stride)
        roi_offset = self.fc(roi_offset_t).reshape(shape=(-1, 2, 7, 7))
        pooled = F.contrib.DeformablePSROIPooling(data=feat, rois=rois, trans=roi_offset,
                                                  group_size=1, pooled_size=7,
                                                  sample_per_part=4, no_trans=False,
                                                  part_size=7, output_dim=self.output_dim,
                                                  spatial_scale=1.0 / self.stride, trans_std=0.1)
        return pooled


class PyramidRFCN(nn.Block):
    def __init__(self, cfg, feature_extractor, *args, **kwargs):
        super(PyramidRFCN, self).__init__(*args, **kwargs)
        self.num_anchors = cfg.network.NUM_ANCHORS
        self.num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else cfg.dataset.NUM_CLASSES)

        self.feature_extractor = feature_extractor
        self.rpn_convs = nn.HybridSequential()
        self.rpn_cls_scores = nn.HybridSequential()
        self.rpn_bbox_preds = nn.HybridSequential()
        # for s in cfg.network.RPN_FEAT_STRIDE:
        self.rpn_conv = nn.Conv2D(channels=512, kernel_size=3, strides=1, padding=1, dilation=1,
                                  activation="relu", use_bias=True, prefix="rpn_conv_")
        self.rpn_cls_score = nn.Conv2D(channels=2 * self.num_anchors, kernel_size=1, strides=1, padding=0, dilation=1,
                                       use_bias=True, prefix="rpn_cls_score_")
        self.rpn_bbox_pred = nn.Conv2D(channels=4 * self.num_anchors, kernel_size=1, strides=1, padding=0, dilation=1,
                                           use_bias=True, prefix="rpn_bbox_pred_")
        #     self.rpn_convs.add(rpn_conv)
        #     self.rpn_cls_scores.add(rpn_cls_score)
        #     self.rpn_bbox_preds.add(rpn_bbox_pred)
        self.cfg = cfg
        self.fpn_pooling_layers = nn.Sequential(prefix="offset_")
        for s in self.cfg.network.RCNN_FEAT_STRIDE:
            base2_stride = int(np.log2(float(s)))
            p = FPNDeformablePSROIPooling(stride=s, prefix="offset_p{}_".format(base2_stride))
            self.fpn_pooling_layers.add(p)

        # Two fc before class classification and bounding box regression
        self.fc_new_1 = nn.Dense(1024, activation="relu", use_bias=True, flatten=True, prefix="fc_new_1_")
        self.fc_new_2 = nn.Dense(1024, activation="relu", use_bias=True, flatten=True, prefix="fc_new_2_")

        # Two fc for class classification and bounding box regression
        self.fc_cls_score = nn.Dense(cfg.dataset.NUM_CLASSES, prefix='cls_score_')
        self.fc_bbox_pred = nn.Dense(self.num_reg_classes * 4, prefix='bbox_pred_')

    def forward(self, x, im_info):
        rois, rpn_cls_scores, rpn_bbox_preds, fpn_features = self.rpn(x, im_info)
        rois_reordered, cls_pred, bbox_pred = self.rcnn(rois, fpn_features)
        cls_pred = cls_pred.softmax(axis=2)
        return rois_reordered, cls_pred, bbox_pred

    def rpn(self, x, im_info):
        assert x.shape[0] == 1, "Only bs==1 is supported."
        fpn_features = self.feature_extractor(x)
        rpn_cls_scores = []
        rpn_bbox_preds = []
        for fpn_feature in fpn_features:
            rpn_relu = self.rpn_conv(fpn_feature)
            rpn_cls_scores.append(self.rpn_cls_score(rpn_relu))
            rpn_bbox_preds.append(self.rpn_bbox_pred(rpn_relu))
        num_anchors = self.cfg.network.NUM_ANCHORS
        rois = pyramid_proposal(
            [x.reshape(0, 2, -1, 0).softmax(axis=1).reshape(0, 2 * num_anchors, -1, 0)[:, num_anchors:].asnumpy() for x
             in rpn_cls_scores],
            [x.asnumpy() for x in rpn_bbox_preds],
            im_info.asnumpy(), self.cfg)
        return rois, rpn_cls_scores, rpn_bbox_preds, fpn_features

    def rcnn(self, rois, fpn_features):
        bs = fpn_features[0].shape[0]
        assert bs == 1, "Only bs==1 is supported."
        w = rois[:, 3] - rois[:, 1] + 1
        h = rois[:, 4] - rois[:, 2] + 1
        feat_id = np.clip(np.floor(2 + np.log2(np.sqrt(w * h) / 224)), 0, len(self.cfg.network.RCNN_FEAT_STRIDE) - 1)
        feature_pooled = []
        keeps = []
        for n, (stride, pooling_layer) in enumerate(zip(self.cfg.network.RCNN_FEAT_STRIDE, self.fpn_pooling_layers)):
            keep = np.where(feat_id == n)[0]
            if len(keep) > 0:
                rois_k = nd.array(rois[keep], ctx=fpn_features[0].context)
                pooled = pooling_layer(rois_k, fpn_features[n])
                feature_pooled.append(pooled)
                keeps.append(keep)
            else:
                # To make mxnet happy
                pooled = pooling_layer(nd.zeros(shape=(1, 5), ctx=fpn_features[0].context), fpn_features[n])
        # Keep the order of feature_pooled same as rois.
        keeps = np.concatenate(keeps, axis=0)
        reverse_index = np.empty(shape=(len(feat_id), ))
        reverse_index[keeps] = range(len(feat_id))
        feature_pooled = nd.concat(*feature_pooled, dim=0)[reverse_index]

        fc_new_1 = self.fc_new_1(feature_pooled)
        fc_new_2 = self.fc_new_2(fc_new_1)
        cls_pred = self.fc_cls_score(fc_new_2).reshape(bs, -1, self.cfg.dataset.NUM_CLASSES)
        bbox_pred = self.fc_bbox_pred(fc_new_2).reshape(bs, -1, 4 * self.num_reg_classes)
        return rois, cls_pred, bbox_pred
