from __future__ import print_function

import logging
import os
import sys
import pprint

import cv2
import mxnet as mx
import mxnet.autograd as ag
import mxnet.ndarray as nd
import numpy as np
import tqdm
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from models.fpn.fpn_dcn import PyramidRFCN
from models.fpn.resnetv1b import ResNetV1, RFPResNetV1
from utils.common import log_init
from utils.config import config, update_config
# from utils.dataloader import DataLoader
from utils.im_detect import im_detect_bbox_aug
from utils.lrsheduler import WarmupMultiFactorScheduler
from utils.parallel import DataParallelModel
from utils.proposal_target import proposal_target
sys.path.append(os.path.join(os.path.dirname(__file__), "../MobulaOP"))

# metric
class RPNAccuMetric(mx.metric.EvalMetric):
    def __init__(self, name="rpn_accu"):
        super(RPNAccuMetric, self).__init__(name)
        self.name = name
        self.preds = []

    def update(self, rpn_cls_label, pred_rpn_box_cls):
        pred_rpn_box_argmax = pred_rpn_box_cls.reshape(2, -1).argmax(axis=0)
        rpn_cls_label = rpn_cls_label.reshape(-1)
        mask = (rpn_cls_label != -1).astype('f')
        rpn_accu = mx.nd.sum(mask * (pred_rpn_box_argmax == rpn_cls_label).astype('f')) / mx.nd.sum(mask)
        self.preds.append(rpn_accu.asscalar())

    def get(self):
        if len(self.preds) > 0:
            return self.name, sum(self.preds) / len(self.preds)
        else:
            return self.name, np.nan

    def reset(self):
        self.preds = []


class RPNCriterion(mx.gluon.nn.Block):
    def forward(self, pred_rpn_box_cls, pred_rpn_box_bbox_deltas, rpn_cls_label, rpn_bbox_target, rpn_bbox_weight):
        pred_rpn_box_cls_reshape = pred_rpn_box_cls.reshape(2, -1).log_softmax(axis=0)
        rpn_cls_label_reshape = rpn_cls_label.reshape(-1)
        loss_rpn_cls_withoutmask = mx.nd.pick(pred_rpn_box_cls_reshape, index=rpn_cls_label_reshape, axis=0)
        mask = (rpn_cls_label_reshape != -1).astype('f')
        grad_scale = 1.0 / mx.nd.sum(mask)
        loss_rpn_cls_masked = -1 * grad_scale * mask * loss_rpn_cls_withoutmask
        loss_rpn_loc = grad_scale * rpn_bbox_weight * mx.nd.smooth_l1(pred_rpn_box_bbox_deltas - rpn_bbox_target,
                                                                      scalar=3.0)
        return loss_rpn_cls_masked.sum(), loss_rpn_loc.sum()


class RCNNCriterion(mx.gluon.nn.Block):
    def forward(self, pred_box_cls, pred_bbox_deltas, cls_labels, bbox_target, bbox_weight):
        pred_box_cls_reshape = pred_box_cls.squeeze().log_softmax(axis=1)  # n x num_classes
        cls_labels_reshape = cls_labels.reshape(-1)
        loss_cls_withoutmask = -1 * mx.nd.pick(pred_box_cls_reshape, index=cls_labels_reshape, axis=1)
        loss_cls_masked = loss_cls_withoutmask
        loss_loc = bbox_weight * mx.nd.smooth_l1(pred_bbox_deltas.squeeze() - bbox_target, scalar=1.0) * (
                    cls_labels > 0).astype('f').expand_dims(axis=1)
        # OHEM
        loss_cls_loc = loss_cls_masked + loss_loc.reshape(0, -1).sum(axis=1)
        ohem_keep = mx.nd.argsort(-1 * loss_cls_loc)[:config.TRAIN.BATCH_ROIS_OHEM]
        grad_scale = 1.0 / len(ohem_keep)
        return grad_scale * loss_cls_masked[ohem_keep].sum(), grad_scale * loss_loc[ohem_keep].sum()


class RCNNWithCriterion(mx.gluon.nn.Block):
    def __init__(self, base_net):
        super(RCNNWithCriterion, self).__init__()
        self.base_net = base_net
        self.rpn_criterion = RPNCriterion()
        self.rcnn_criterion = RCNNCriterion()

        self.cfg = config

    def forward(self, data, im_info, gt_boxes, rpn_label, rpn_bbox_target, rpn_bbox_weight):
        # We use a 1x1 tensor to indicate there is no gt_boxes for this images, as a zero size tensor can not be passed
        # through multi-processing Dataloader (and also as_in_context).
        is_negative = gt_boxes.size < 2
        if not is_negative:
            gt_boxes = gt_boxes[:, :, :5]
        rois, rpn_cls_scores, rpn_bbox_preds, fpn_features = self.base_net.rpn(data, im_info)
        if not is_negative:
            proposal_targets_r = proposal_target(rois, gt_boxes[0].asnumpy(), config)
            rcnn_labels, rcnn_bbox_targets, rcnn_bbox_weights = (nd.array(x, ctx=data.context) for x in
                                                                 proposal_targets_r[1:])
            rois = proposal_targets_r[0]
        rois, rcnn_cls_score, rcnn_bbox_pred = self.base_net.rcnn(rois, fpn_features)
        # If there is no gt_boxes for one image, we call it negative sample. Bounding box regression is not needed for
        # a negative sample, so the rcnn_bbox_targets and rcnn_bbox_weights are both all zeros, and so is rcnn_labels.
        if is_negative:
            rcnn_bbox_targets = nd.zeros_like(rcnn_bbox_pred.squeeze())
            rcnn_bbox_weights = nd.zeros_like(rcnn_bbox_pred.squeeze())
            rcnn_labels = nd.zeros(shape=(rois.shape[0],), dtype='f', ctx=data.context)

        rpn_cls_scores = [x.reshape(0, 2, -1) for x in rpn_cls_scores]
        rpn_cls_scores = mx.nd.concat(*rpn_cls_scores, dim=2)
        rpn_bbox_preds = [x.reshape(1, 4 * config.network.NUM_ANCHORS, -1) for x in rpn_bbox_preds]
        rpn_bbox_preds = mx.nd.concat(*rpn_bbox_preds, dim=2)

        loss_rpn_cls, loss_rpn_loc = self.rpn_criterion(rpn_cls_scores, rpn_bbox_preds,
                                                        rpn_label, rpn_bbox_target, rpn_bbox_weight)
        loss_rcnn_cls, loss_rcnn_loc = self.rcnn_criterion(rcnn_cls_score, rcnn_bbox_pred, rcnn_labels,
                                                           rcnn_bbox_targets,
                                                           rcnn_bbox_weights)
        return loss_rpn_cls, loss_rpn_loc, loss_rcnn_cls, loss_rcnn_loc, rpn_label, rpn_cls_scores


def batch_fn(x):
    return x


def train_net(ctx, begin_epoch, lr, lr_step):
    mx.random.seed(3)
    np.random.seed(3)

    batch_size = len(ctx)
    if config.network.USE_RFP:
        backbone = RFPResNetV1(num_devices=len(set(ctx)), num_layers=50, sync_bn=config.network.SYNC_BN, pretrained=True)
    else:
        backbone = ResNetV1(num_devices=len(set(ctx)), num_layers=50, sync_bn=config.network.SYNC_BN, pretrained=True)
    feat_symbol = backbone(mx.sym.var(name="data"))
    net = PyramidRFCN(config, backbone)

    # Resume parameters.
    resume = None
    if resume is not None:
        params_coco = mx.nd.load(resume)
        for k in params_coco:
            params_coco[k.replace("arg:", "").replace("aux:", "")] = params_coco.pop(k)
        params = net.collect_params()

        for k in params.keys():
            try:
                params[k]._load_init(params_coco[k], ctx=mx.cpu())
            except Exception as e:
                logging.exception(e)

    # Initialize parameters
    params = net.collect_params()
    for key in params.keys():
        if params[key]._data is None:
            default_init = mx.init.Zero() if "bias" in key or "offset" in key else mx.init.Normal()
            default_init.set_verbosity(True)
            if params[key].init is not None and hasattr(params[key].init, "set_verbosity"):
                params[key].init.set_verbosity(True)
                params[key].initialize(init=params[key].init, default_init=params[key].init)
            else:
                params[key].initialize(default_init=default_init)

    net.collect_params().reset_ctx(list(set(ctx)))
    import data.transforms.bbox as bbox_t
    train_transforms = bbox_t.Compose([
        # Flipping is implemented in dataset.
        # bbox_t.RandomRotate(bound=True, min_angle=-15, max_angle=15),
        bbox_t.Resize(target_size=config.SCALES[0][0], max_size=config.SCALES[0][1]),
        # bbox_t.RandomResize(scales=[(960, 2000), (800, 1600), (600, 1200)]),
        bbox_t.Normalize(),
        bbox_t.AssignPyramidAnchor(config, symbol=feat_symbol, pad_n=32)
    ])
    val_transforms = bbox_t.Compose([
        bbox_t.Resize(target_size=config.SCALES[0][0], max_size=config.SCALES[0][1]),
        bbox_t.Normalize(),
    ])
    from data.bbox.mscoco import COCODetection
    val_dataset = COCODetection(root=config.dataset.dataset_path, splits=("instances_val2017",), h_flip=False)
    train_dataset = COCODetection(root=config.dataset.dataset_path, splits=("instances_train2017",),
                                  h_flip=config.TRAIN.FLIP,
                                  transform=train_transforms)
    # val_dataset = YunChongDataSet(is_train=False, h_flip=False)

    # train_loader = DataLoader(train_dataset, batchsize=len(ctx))
    train_loader = mx.gluon.data.DataLoader(dataset=train_dataset, batch_size=len(ctx), batchify_fn=batch_fn,
                                            pin_memory=False, num_workers=0, last_batch="discard", shuffle=True)
    # for _ in tqdm.tqdm(train_loader, desc="Checking Dataset"):
    #     pass

    rpn_eval_metric = RPNAccuMetric()
    loss_rpn_cls_metric = mx.metric.Loss(name="rpn_cls")
    loss_rpn_loc_metric = mx.metric.Loss(name="rpn_loc")
    loss_rcnn_cls_metric = mx.metric.Loss(name="rcnn_cls")
    loss_rcnn_loc_metric = mx.metric.Loss(name="rcnn_loc")

    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [rpn_eval_metric, loss_rpn_cls_metric, loss_rpn_loc_metric, loss_rcnn_cls_metric,
                         loss_rcnn_loc_metric]:
        eval_metrics.add(child_metric)

    params_all = net.collect_params()
    params_to_train = {}
    params_fixed_prefix = config.network.FIXED_PARAMS
    for p in params_all.keys():
        ignore = False
        if params_fixed_prefix is not None:
            for f in params_fixed_prefix:
                if f in str(p):
                    ignore = True
                    params_all[p].grad_req = 'null'
                    logging.info("{} is ignored when training.".format(p))
        if not ignore: params_to_train[p] = params_all[p]
    base_lr = lr
    lr_factor = config.TRAIN.lr_factor
    lr_epoch = [float(epoch) for epoch in lr_step.split(',')]
    lr_epoch_diff = [epoch - begin_epoch for epoch in lr_epoch if epoch > begin_epoch]
    lr = base_lr * (lr_factor ** (len(lr_epoch) - len(lr_epoch_diff)))
    lr_iters = [int(epoch * len(train_dataset) / batch_size) for epoch in lr_epoch_diff]
    print('lr', lr, 'lr_epoch_diff', lr_epoch_diff, 'lr_iters', lr_iters)
    lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, config.TRAIN.warmup, config.TRAIN.warmup_lr,
                                              config.TRAIN.warmup_step)

    trainer = mx.gluon.Trainer(
        params_to_train,  # fix batchnorm, fix first stage, etc...
        'sgd',
        {'wd': config.TRAIN.wd,
         'momentum': config.TRAIN.momentum,
         'clip_gradient': None,
         'lr_scheduler': lr_scheduler
         })
    val_metric_5 = VOC07MApMetric(iou_thresh=.5)

    net_with_criterion = RCNNWithCriterion(base_net=net)
    net_parallel = DataParallelModel(net_with_criterion, ctx_list=ctx,
                                     sync=True if config.network.IM_PER_GPU is 1 else False)

    for epoch in range(begin_epoch, config.TRAIN.end_epoch):
        eval_metrics.reset()
        net.feature_extractor.hybridize(static_alloc=True, static_shape=False)
        _ = net(mx.random.randn(1, 3, 512, 512, ctx=ctx[0]), mx.nd.array([[512, 512, 1]], ctx=ctx[0]))
        for nbatch, data_batch in enumerate(tqdm.tqdm(train_loader, total=len(train_dataset) // batch_size,
                                                      unit_scale=batch_size)):
            inputs = [[x.as_in_context(c) for x in d] for c, d in zip(ctx, data_batch)]
            losses = []
            net.collect_params().zero_grad()
            with ag.record():
                outputs = net_parallel(*inputs)
            for output in outputs:
                loss_rpn_cls, loss_rpn_loc, loss_rcnn_cls, loss_rcnn_loc, rpn_label, rpn_cls_score = output
                losses.extend([loss_rpn_cls, loss_rpn_loc, loss_rcnn_cls, loss_rcnn_loc])
            rpn_eval_metric.update(rpn_label, rpn_cls_score)
            loss_rpn_cls_metric.update(None, loss_rpn_cls)
            loss_rpn_loc_metric.update(None, loss_rpn_loc)
            loss_rcnn_cls_metric.update(None, loss_rcnn_cls)
            loss_rcnn_loc_metric.update(None, loss_rcnn_loc)
            ag.backward(losses)
            trainer.step(len(ctx), ignore_stale_grad=True)
            mx.nd.waitall()
            if nbatch % 100 == 0:
                msg = ','.join(['{}={:.3f}'.format(w, v) for w, v in zip(*eval_metrics.get())])
                msg += ",lr={}".format(trainer.learning_rate)
                logging.info(msg)
                rpn_eval_metric.reset()
            if nbatch % 10000 ==0:
                save_path = "{}-{}-{}.params".format(config.TRAIN.model_prefix, epoch, nbatch)
                net.collect_params().save(save_path)
                trainer.save_states(config.TRAIN.model_prefix + "-trainer.states")
                logging.info("Saved checkpoint to {}.".format(save_path))
        # val_metric_5.reset()
        # for i in tqdm.tqdm(range(len(val_dataset))):
        #     img_path, gt_boxes = val_dataset.at_with_image_path(i)
        #     pred_bboxes, pred_scores, pred_clsid = im_detect_bbox_aug(net, nms_threshold=config.TEST.NMS,
        #                                                               im=cv2.imread(img_path)[:, :, ::-1],
        #                                                               scales=config.SCALES,
        #                                                               ctx=ctx,
        #                                                               bbox_stds=config.TRAIN.BBOX_STDS,
        #                                                               flip=True,
        #                                                               threshold=1e-3,
        #                                                               viz=False,
        #                                                               pad=32,
        #                                                               class_agnostic=config.CLASS_AGNOSTIC
        #                                                               )
        #     val_metric_5.update(pred_bboxes=pred_bboxes[np.newaxis],
        #                         pred_labels=pred_clsid[np.newaxis] - 1,
        #                         pred_scores=pred_scores[np.newaxis],
        #                         gt_bboxes=gt_boxes[np.newaxis, :, :4],
        #                         gt_labels=gt_boxes[np.newaxis, :, 4],
        #                         gt_difficults=gt_boxes[np.newaxis, :, 5])
        # re = val_metric_5.get()
        re = ("mAP", "0.0")
        logging.info(re)
        save_path = "{}-{}-{}.params".format(config.TRAIN.model_prefix, epoch, re[1])
        net.collect_params().save(save_path)
        trainer.save_states(config.TRAIN.model_prefix + "-trainer.states")
        logging.info("Saved checkpoint to {}.".format(save_path))


def main():
    update_config("configs/coco/resnet_v1_101_coco_trainval_fpn_dcn_end2end_ohem.yaml")
    os.makedirs(config.TRAIN.model_prefix, exist_ok=True)
    log_init(filename=config.TRAIN.model_prefix + "train.log")
    msg = pprint.pformat(config)
    logging.info(msg)
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    os.environ["MXNET_GPU_MEM_POOL_TYPE"] = "Round"

    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    ctx = ctx * config.network.IM_PER_GPU
    train_net(ctx, config.TRAIN.begin_epoch, config.TRAIN.lr, config.TRAIN.lr_step)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
