import mxnet as mx
import mxnet.gluon.nn as nn


class FasterRCNN(nn.Block):
    def __init__(self, cfg, rpn_rcnn_feat, *args, **kwargs):
        super(FasterRCNN, self).__init__(*args, **kwargs)
        self.cfg = cfg
        with self.name_scope():
            # Backbone
            self.rpn_rcnn_feat = rpn_rcnn_feat

            # RPN
            self.rpn_relu = nn.Conv2D(weight_initializer=mx.init.Normal(),bias_initializer=mx.init.Zero(), channels=512, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1), activation="relu", prefix="rpn_conv_3x3_")
            self.rpn_cls_score = nn.Conv2D(weight_initializer=mx.init.Normal(),bias_initializer=mx.init.Zero(), channels=2 * cfg.network.NUM_ANCHORS, kernel_size=(1, 1), strides=(1, 1), prefix="rpn_cls_score_")
            self.rpn_bbox_pred = nn.Conv2D(weight_initializer=mx.init.Normal(),bias_initializer=mx.init.Zero(), channels=4 * cfg.network.NUM_ANCHORS, kernel_size=(1, 1), strides=(1, 1), prefix="rpn_bbox_pred_")

            # RCNN
            self.relu_new_1 = nn.Conv2D(weight_initializer=mx.init.Normal(),bias_initializer=mx.init.Zero(), channels=1024, kernel_size=(1, 1), strides=(1, 1), activation="relu", prefix="conv_new_1_")
            self.relu_new_1.collect_params().setattr('lr_mult', 3)
            self.rfcn_cls = nn.Conv2D(weight_initializer=mx.init.Normal(),bias_initializer=mx.init.Zero(), channels=7 * 7 * cfg.dataset.NUM_CLASSES, kernel_size=(1, 1), strides=(1, 1), prefix="rfcn_cls_")
            self.rfcn_bbox = nn.Conv2D(weight_initializer=mx.init.Normal(),bias_initializer=mx.init.Zero(), channels=7 * 7 * (2 if cfg.CLASS_AGNOSTIC else cfg.dataset.NUM_CLASSES) * 4, kernel_size=(1, 1), strides=(1, 1), prefix="rfcn_bbox_")
            self.rfcn_cls_offset_t = nn.Conv2D(weight_initializer=mx.init.Zero(), bias_initializer=mx.init.Zero(), channels=2 * 7 * 7 * cfg.dataset.NUM_CLASSES, kernel_size=(1, 1), strides=(1, 1), prefix="rfcn_cls_offset_t_")
            self.rfcn_bbox_offset_t = nn.Conv2D(weight_initializer=mx.init.Zero(), bias_initializer=mx.init.Zero(), channels=2 * 7 * 7, kernel_size=(1, 1), strides=(1, 1), prefix="rfcn_bbox_offset_t_")

    def forward(self, data, im_info):
        rpn_feat, rcnn_feat = self.feat(data)
        rois, rpn_cls_score, rpn_bbox_pred = self.rpn(rpn_feat, im_info)
        rcnn_cls_score, rcnn_bbox_pred = self.rcnn(rois, rcnn_feat)
        scores = rcnn_cls_score.reshape((1, -1, self.cfg.dataset.NUM_CLASSES)).softmax(axis = 2)
        bbox_deltas = rcnn_bbox_pred.reshape((1, -1, 4 * (2 if self.cfg.CLASS_AGNOSTIC else self.cfg.dataset.NUM_CLASSES)))
        return rois, scores, bbox_deltas

    def feat(self, data):
        rpn_feat, rcnn_feat = self.rpn_rcnn_feat(data)
        return rpn_feat, rcnn_feat

    def rpn(self, rpn_feat, im_info):
        # RPN
        rpn_relu = self.rpn_relu(rpn_feat)
        rpn_cls_score = self.rpn_cls_score(rpn_relu)
        rpn_bbox_pred = self.rpn_bbox_pred(rpn_relu)
        rois = self.proposal(rpn_cls_score, rpn_bbox_pred, im_info)
        return rois, rpn_cls_score, rpn_bbox_pred

    def rcnn(self, rois, rcnn_feat):
        cfg = self.cfg
        num_classes = cfg.dataset.NUM_CLASSES
        num_reg_classes = (2 if cfg.CLASS_AGNOSTIC else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS


        # RCNN
        relu_new_1 = self.relu_new_1(rcnn_feat)
        rfcn_cls = self.rfcn_cls(relu_new_1)
        rfcn_bbox = self.rfcn_bbox(relu_new_1)

        # Deformable PSROIPooling
        rfcn_cls_offset_t = self.rfcn_cls_offset_t(relu_new_1)
        rfcn_bbox_offset_t = self.rfcn_bbox_offset_t(relu_new_1)
        rfcn_cls_offset = mx.nd.contrib.DeformablePSROIPooling(data=rfcn_cls_offset_t, rois=rois, group_size=7, pooled_size=7, sample_per_part=4, no_trans=True, part_size=7, output_dim=2 * num_classes, spatial_scale=1.0 / self.cfg.network.RPN_FEAT_STRIDE)
        rfcn_bbox_offset = mx.nd.contrib.DeformablePSROIPooling(data=rfcn_bbox_offset_t, rois=rois, group_size=7, pooled_size=7, sample_per_part=4, no_trans=True, part_size=7, output_dim=2, spatial_scale=1.0 / self.cfg.network.RPN_FEAT_STRIDE)
        psroipooled_cls_rois = mx.nd.contrib.DeformablePSROIPooling(data=rfcn_cls, rois=rois, trans=rfcn_cls_offset, group_size=7, pooled_size=7, sample_per_part=4, no_trans=False, trans_std=0.1, output_dim=num_classes, spatial_scale=1.0 / self.cfg.network.RPN_FEAT_STRIDE, part_size=7)
        psroipooled_loc_rois = mx.nd.contrib.DeformablePSROIPooling(data=rfcn_bbox, rois=rois, trans=rfcn_bbox_offset, group_size=7, pooled_size=7, sample_per_part=4, no_trans=False, trans_std=0.1, output_dim=8, spatial_scale=1.0 / self.cfg.network.RPN_FEAT_STRIDE, part_size=7)
        rcnn_cls_score = mx.nd.Pooling(data=psroipooled_cls_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        rcnn_bbox_pred = mx.nd.Pooling(data=psroipooled_loc_rois, pool_type='avg', global_pool=True, kernel=(7, 7))
        return rcnn_cls_score, rcnn_bbox_pred

    def proposal(self, rpn_cls_score, rpn_bbox_pred, im_info):
        cfg = self.cfg
        rpn_cls_score = mx.nd.reshape(rpn_cls_score, shape=(0, 2, -1, 0)).softmax(axis = 1).reshape(shape = rpn_cls_score.shape)
        rois = mx.nd.contrib.Proposal(
            cls_prob=rpn_cls_score, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
            feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
            ratios=tuple(cfg.network.ANCHOR_RATIOS),
            rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
            threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
        return rois
