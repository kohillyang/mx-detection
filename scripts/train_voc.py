from __future__ import print_function

import logging
import os
import pprint

import cv2
import mxnet as mx
import mxnet.ndarray as nd
import mxnet.autograd as ag
import numpy as np
import tqdm
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric

from data.bbox.voc import VOCDetection
from models.rfcn.gluoncv_resnet import ResNetV1
from models.rfcn.rfcn import FasterRCNN
from utils.common import log_init
from utils.config import config, update_config
from utils.lrsheduler import WarmupMultiFactorScheduler
from utils.parallel import DataParallelModel
from utils.proposal_target import proposal_target
from utils.im_detect import im_detect_bbox_aug
from utils.dataloader import DataLoader


# metric
class RPNAccuMetric(mx.metric.EvalMetric):
    def __init__(self, name = "rpn_accu"):
        super(RPNAccuMetric, self).__init__(name)
        self.name = name
        self.preds = []

    def update(self, rpn_cls_label, pred_rpn_box_cls):
        with ag.pause():
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
        pred_rpn_box_cls_reshape = pred_rpn_box_cls.reshape(2, -1).log_softmax(axis = 0)
        rpn_cls_label_reshape = rpn_cls_label.reshape(-1)
        loss_rpn_cls_withoutmask = mx.nd.pick(pred_rpn_box_cls_reshape, index=rpn_cls_label_reshape, axis=0)
        mask = (rpn_cls_label_reshape != -1).astype('f')
        grad_scale = 1.0 / mx.nd.sum(mask)
        loss_rpn_cls_masked = -1 * grad_scale * mask * loss_rpn_cls_withoutmask
        loss_rpn_loc = grad_scale * rpn_bbox_weight * mx.nd.smooth_l1(pred_rpn_box_bbox_deltas - rpn_bbox_target, scalar=3.0)
        return loss_rpn_cls_masked.sum(), loss_rpn_loc.sum()


class RCNNCriterion(mx.gluon.nn.Block):
    def forward(self, pred_box_cls, pred_bbox_deltas, cls_labels, bbox_target, bbox_weight):
        pred_box_cls_reshape = pred_box_cls.squeeze().log_softmax(axis=1) # n x num_classes
        cls_labels_reshape = cls_labels.reshape(-1)
        loss_cls_withoutmask = -1 * mx.nd.pick(pred_box_cls_reshape, index=cls_labels_reshape, axis=1)
        loss_cls_masked = loss_cls_withoutmask
        loss_loc = bbox_weight * mx.nd.smooth_l1(pred_bbox_deltas.squeeze() - bbox_target, scalar=3.0) * (cls_labels > 0).astype('f').expand_dims(axis=1)
        # OHEM
        with ag.pause():
            loss_cls_loc = loss_cls_masked + loss_loc.reshape(0,-1).sum(axis=1)
            ohem_keep = mx.nd.argsort(-1 * loss_cls_loc)[:config.TRAIN.BATCH_ROIS_OHEM]
            grad_scale = 1.0 / len(ohem_keep)
        return grad_scale * loss_cls_masked[ohem_keep].sum(), grad_scale * loss_loc[ohem_keep].sum()


class RCNNWithCriterion(mx.gluon.nn.Block):
    def __init__(self, base_net):
        super(RCNNWithCriterion, self).__init__()
        self.base_net = base_net
        self.rpn_criterion = RPNCriterion()
        self.rcnn_criterion = RCNNCriterion()

    def forward(self, data, im_info, gt_boxes, rpn_label, rpn_bbox_target, rpn_bbox_weight):
        rpn_feat, rcnn_feat = self.base_net.feat(data)
        rois, rpn_cls_score, rpn_bbox_pred = self.base_net.rpn(rpn_feat, im_info)
        proposal_targets_r = proposal_target(rois.asnumpy(), gt_boxes[0].asnumpy(), config)
        rois, rcnn_labels, rcnn_bbox_targets, rcnn_bbox_weights = (nd.array(x, ctx=rois.context) for x in
                                                                   proposal_targets_r)
        rcnn_cls_score, rcnn_bbox_pred = self.base_net.rcnn(rois, rcnn_feat)
        loss_rpn_cls, loss_rpn_loc = self.rpn_criterion(rpn_cls_score, rpn_bbox_pred, rpn_label, rpn_bbox_target,
                                                   rpn_bbox_weight)
        loss_rcnn_cls, loss_rcnn_loc = self.rcnn_criterion(rcnn_cls_score, rcnn_bbox_pred, rcnn_labels, rcnn_bbox_targets,
                                                      rcnn_bbox_weights)
        return loss_rpn_cls, loss_rpn_loc, loss_rcnn_cls, loss_rcnn_loc, rpn_label, rpn_cls_score


def train_net(ctx, pretrained, epoch, prefix, begin_epoch, end_epoch, lr, lr_step):
    mx.random.seed(3)
    np.random.seed(3)

    batch_size = len(ctx)
    backbone = ResNetV1()
    feat_symbol = backbone(mx.symbol.var(name="data"))[0]
    net = FasterRCNN(config, backbone)

    params = net.collect_params()
    params_pretrained = None #
    # uncommit the following line to load pretrained model.
    # params_pretrained = mx.nd.load("pretrained/rfcn-voc-resnet50_v1--29-0.804082102562.params")
    if params_pretrained is not None:
        for k in params.keys():
            try:
                params[k]._load_init(params_pretrained[k], mx.cpu())
            except Exception as e:
                logging.exception(e)
    for key in params.keys():
        if params[key]._data is None:
            default_init = mx.init.Zero() if "bias" in key or "offset" in key else mx.init.Normal()
            default_init.set_verbosity(True)
            if params[key].init is not None:
                params[key].init.set_verbosity(True)
                params[key].initialize(init=params[key].init, default_init = params[key].init)
            else:
                params[key].initialize(default_init=default_init)
    net.collect_params().reset_ctx(list(set(ctx)))
    import data.transforms.bbox as bbox_t
    train_transforms = bbox_t.Compose([
        # bbox_t.RandomRotate(bound=True, min_angle=-15, max_angle=15),
        bbox_t.Resize(target_size=config.SCALES[0][0], max_size=config.SCALES[0][1]),
        bbox_t.Normalize(),
        bbox_t.AssignAnchor(config, feat_strides=(16, 16), symbol=feat_symbol)
    ])
    val_transforms = bbox_t.Compose([
        bbox_t.Resize(target_size=config.SCALES[0][0], max_size=config.SCALES[0][1]),
        bbox_t.Normalize(),
    ])

    train_dataset = VOCDetection(root=config.dataset.dataset_path, splits=((2007, 'trainval'), (2012, 'trainval')), transform=train_transforms)
    val_dataset = VOCDetection(root=config.dataset.dataset_path, splits=((2007, 'test'),))

    train_loader = DataLoader(train_dataset, batchsize=len(ctx))
    # train_loader = mx.gluon.data.DataLoader(dataset=train_dataset, batch_size=len(ctx), batchify_fn=lambda x: x,
    #                                         pin_memory=True, num_workers=8, last_batch="discard")

    rpn_eval_metric = RPNAccuMetric()
    loss_rpn_cls_metric = mx.metric.Loss(name="rpn_cls")
    loss_rpn_loc_metric = mx.metric.Loss(name="rpn_loc")
    loss_rcnn_cls_metric = mx.metric.Loss(name="rcnn_cls")
    loss_rcnn_loc_metric = mx.metric.Loss(name="rcnn_loc")

    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [rpn_eval_metric, loss_rpn_cls_metric, loss_rpn_loc_metric, loss_rcnn_cls_metric, loss_rcnn_loc_metric]:
        eval_metrics.add(child_metric)

    params_all = net.collect_params()
    params_to_train = {}
    params_fixed_prefix = config.network.FIXED_PARAMS
    for p in params_all.keys():
        ignore = False
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
    lr_scheduler = WarmupMultiFactorScheduler(lr_iters, lr_factor, config.TRAIN.warmup, config.TRAIN.warmup_lr, config.TRAIN.warmup_step)

    trainer = mx.gluon.Trainer(
        net.collect_params(),  # fix batchnorm, fix first stage, etc...
        'sgd',
        {'learning_rate': config.TRAIN.lr,
         'wd': config.TRAIN.wd,
         'momentum': config.TRAIN.momentum,
         'clip_gradient': None,
         'lr_scheduler': lr_scheduler
         })
    val_metric_5 = VOC07MApMetric(iou_thresh=.5)

    net_with_criterion = RCNNWithCriterion(base_net=net)
    net_parallel = DataParallelModel(net_with_criterion, ctx_list=ctx, sync=True)

    for epoch in range(begin_epoch, config.TRAIN.end_epoch):
        # train_data.reset()
        net.hybridize(static_alloc=True, static_shape=False)
        _ = net(mx.random.randn(1,3,512,512, ctx=ctx[0]), mx.nd.array([[512,512,1]], ctx=ctx[0]))
        for nbatch, data_batch in enumerate(tqdm.tqdm(train_loader, total = len(train_dataset) // batch_size)):
            inputs = [[x.as_in_context(c) for x in d] for c, d in zip(ctx, data_batch)]
            losses = []
            with ag.record():
                outputs = net_parallel(*inputs)
                for output in outputs:
                    loss_rpn_cls, loss_rpn_loc, loss_rcnn_cls, loss_rcnn_loc, rpn_label, rpn_cls_score = output
                    if nbatch % 4 == 0:
                        rpn_eval_metric.update(rpn_label, rpn_cls_score)
                        loss_rpn_cls_metric.update(None, loss_rpn_cls)
                        loss_rpn_loc_metric.update(None, loss_rpn_loc)
                        loss_rcnn_cls_metric.update(None, loss_rcnn_cls)
                        loss_rcnn_loc_metric.update(None, loss_rcnn_loc)
                    losses.extend([loss_rpn_cls, loss_rpn_loc, loss_rcnn_cls, loss_rcnn_loc])
            ag.backward(losses)
            trainer.step(1, ignore_stale_grad=True)
            if nbatch % 100 == 0:
                msg = ','.join(['{}={:.3f}'.format(w,v) for w,v in zip(*eval_metrics.get())])
                msg += ",lr={}".format(trainer.learning_rate)
                logging.info(msg)
                rpn_eval_metric.reset()
        val_metric_5.reset()
        net.hybridize(static_alloc=True, static_shape=False)
        for i in tqdm.tqdm(range(len(val_dataset))):
            img_path, gt_boxes = val_dataset.at_with_image_path(i)
            pred_bboxes, pred_scores, pred_clsid = im_detect_bbox_aug(net,nms_threshold=config.TEST.NMS,
                                                                      im=cv2.imread(img_path)[:, :, ::-1], # bgr
                                                                      scales=config.SCALES,
                                                                      ctx=ctx,
                                                                      bbox_stds=config.TRAIN.BBOX_STDS,
                                                                      threshold=1e-3,
                                                                      viz=False
                                                                      )
            val_metric_5.update(pred_bboxes = pred_bboxes[np.newaxis],
                                pred_labels = pred_clsid[np.newaxis]-1,
                                pred_scores = pred_scores[np.newaxis],
                                gt_bboxes = gt_boxes[np.newaxis,:,:4],
                                gt_labels = gt_boxes[np.newaxis,:,4],
                                gt_difficults=gt_boxes[np.newaxis,:,5])
        re = val_metric_5.get()
        logging.info(re)
        save_path = "{}-{}-{}.params".format(config.TRAIN.model_prefix,epoch,re[1])
        net.collect_params().save(save_path)
        logging.info("Saved checkpoint to {}.".format(save_path))


def main():
    update_config("configs/voc/resnet_v1_50_voc0712_rfcn_dcn_end2end_ohem_one_gpu.yaml")
    log_init(filename=config.TRAIN.model_prefix + "train.log")
    msg = pprint.pformat(config)
    logging.info(msg)
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    os.environ["MXNET_GPU_MEM_POOL_TYPE"] = "Round"

    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    train_net(ctx, config.network.pretrained, config.network.pretrained_epoch, config.TRAIN.model_prefix,
              config.TRAIN.begin_epoch, config.TRAIN.end_epoch, config.TRAIN.lr,
               config.TRAIN.lr_step)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
