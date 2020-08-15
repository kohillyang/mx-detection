from __future__ import print_function

import logging
import os
import pprint
import sys
import argparse

import cv2
import mxnet as mx
import mxnet.autograd as ag
import numpy as np
import tqdm
import time
import easydict
import gluoncv

import matplotlib.pyplot as plt

from models.retinanet.resnetv1b import FPNResNetV1
from utils.common import log_init
from data.bbox.bbox_dataset import AspectGroupingDataset

sys.path.append(os.path.join(os.path.dirname(__file__), "../MobulaOP"))
import data.transforms.bbox as bbox_t
import mobula
setattr(mobula.config, "NVCC", "/usr/local/cuda-10.0/bin/nvcc")
mobula.op.load('RetinaNetTargetGenerator', os.path.join(os.path.dirname(__file__), "../utils/operator_cxx"))
mobula.op.load('RetinaNetRegression', os.path.join(os.path.dirname(__file__), "../utils/operator_cxx"))


@mobula.op.register
class BCELoss:
    def forward(self, y, target):
        return mx.nd.log(1 + mx.nd.exp(y)) - target * y

    def backward(self, dy):
        grad = mx.nd.sigmoid(self.X[0]) - self.X[1]
        # grad *= 1e-4
        self.dX[0][:] = grad * dy

    def infer_shape(self, in_shape):
        try:
            assert in_shape[0] == in_shape[1]
        except AssertionError as e:
            print(in_shape)
            raise e
        return in_shape, [in_shape[0]]


def BCEFocalLossWithoutAlpha(x, target):
    # p = x.sigmoid()
    # loss = target * ((1-p)**2) * mx.nd.log(p + 1e-7) + (1-target) * (p **2) * mx.nd.log(1 - p + 1e-7)
    # return (-1 * loss).mean()
    bce_loss = BCELoss(x, target)
    pt = mx.nd.exp(-1 * bce_loss)
    r = bce_loss * (1-pt) **2
    return r


def BCEFocalLoss(x, target, alpha=.25, gamma=2):
    alpha = .25
    p = x.sigmoid()
    loss = alpha * target * ((1-p)**2) * mx.nd.log(p + 1e-11)
    loss = loss + (1-alpha) * (1-target) * (p **2) * mx.nd.log(1 - p + 1e-11)
    return -loss


def batch_fn(x):
    return x


class RetinaNet_Head(mx.gluon.nn.HybridBlock):
    def __init__(self, num_classes, num_anchors):
        super(RetinaNet_Head, self).__init__()
        with self.name_scope():
            self.feat_cls = mx.gluon.nn.HybridSequential()
            init = mx.init.Normal(sigma=0.01)
            init.set_verbosity(True)
            init_bias = mx.init.Constant(-1 * np.log((1-0.01) / 0.01))
            init_bias.set_verbosity(True)
            for i in range(4):
                self.feat_cls.add(mx.gluon.nn.Conv2D(channels=256, kernel_size=3, padding=1, weight_initializer=init))
                self.feat_cls.add(mx.gluon.nn.GroupNorm(num_groups=32))
                self.feat_cls.add(mx.gluon.nn.Activation(activation="relu"))
            num_cls_channel = (num_classes - 1) * num_anchors
            self.feat_cls.add(mx.gluon.nn.Conv2D(channels=num_cls_channel, kernel_size=1, padding=0,
                                                 bias_initializer=init_bias, weight_initializer=init))

            self.feat_reg = mx.gluon.nn.HybridSequential()
            for i in range(4):
                self.feat_reg.add(mx.gluon.nn.Conv2D(channels=256, kernel_size=3, padding=1, weight_initializer=init))
                self.feat_reg.add(mx.gluon.nn.GroupNorm(num_groups=32))
                self.feat_reg.add(mx.gluon.nn.Activation(activation="relu"))

            self.feat_reg_loc = mx.gluon.nn.Conv2D(channels=4 * num_anchors, kernel_size=1, padding=0,
                                                   weight_initializer=init)

    def hybrid_forward(self, F, x):
        feat_reg = self.feat_reg(x)
        x_loc = self.feat_reg_loc(feat_reg)
        x_cls = self.feat_cls(x)
        x = F.concat(x_loc, x_cls, dim=1)
        return x


class FCOSFPNNet(mx.gluon.nn.HybridBlock):
    def __init__(self, backbone, num_classes, num_anchors):
        super(FCOSFPNNet, self).__init__()
        self.backbone = backbone
        self._head = RetinaNet_Head(num_classes, num_anchors)

    def hybrid_forward(self, F, x):
        # typically the strides are (4, 8, 16, 32, 64)
        x = self.backbone(x)
        if isinstance(x, list) or isinstance(x, tuple):
            return [self._head(xx) for xx in x]
        else:
            return [self._head(x)]


class RetinaNetTargetGenerator(object):
    def __init__(self, config):
        super(RetinaNetTargetGenerator, self).__init__()
        self.config = config
        self.strides = self.config.retinanet.network.FPN_STRIDES
        self.base_sizes = self.config.retinanet.network.BASE_SIZES
        self.number_of_classes = self.config.dataset.NUM_CLASSES
        self._debug_show_fig = False
        self.bbox_norm_coef = self.config.retinanet.network.bbox_norm_coef

    def __call__(self, image_transposed, bboxes):
        h, w, c = image_transposed.shape
        bboxes = bboxes.copy()
        bboxes[:, 4] += 1
        outputs = [image_transposed]
        if self._debug_show_fig:
            fig, axes = plt.subplots(3, 3)
            axes = axes.reshape(-1)
            n_axes = 0
        targets = []
        for stride, base_size in zip(self.strides, self.base_sizes):
            target = mobula.op.RetinaNetTargetGenerator[np.ndarray](number_of_classes=self.number_of_classes,
                                                                    stride=stride, base_size=base_size)(
                image_transposed.astype(np.float32), bboxes.astype(np.float32))
            target[:, :, :, 2:6] /= np.array(self.bbox_norm_coef)[None, None, None]
            if self._debug_show_fig:
                axes[n_axes].imshow(target[:, :, :, 6:].max(axis=2).max(axis=2))
                n_axes += 1
            # target = np.transpose(target.reshape((target.shape[0], target.shape[1], -1)), (2, 0, 1))
            target = target.transpose((3, 2, 0, 1))
            target = target.reshape((target.shape[0], target.shape[1], -1))  # 6 + no. of classes, num_anchors, -1
            targets.append(target)
        if self._debug_show_fig:
            axes[n_axes].imshow(image_transposed.astype(np.uint8))
            gluoncv.utils.viz.plot_bbox(image_transposed, bboxes=bboxes[:, :4], ax=axes[n_axes])
            plt.show()
        targets = np.concatenate(targets, axis=2)
        outputs.append(targets)
        outputs = tuple(mx.nd.array(x) for x in outputs)
        return outputs


def batch_fn(x):
    return x


def train_net(config):
    mx.random.seed(3)
    np.random.seed(3)

    backbone = FPNResNetV1(sync_bn=config.network.sync_bn, num_devices=len(config.gpus), use_global_stats=config.network.use_global_stats)
    batch_size = config.TRAIN.batch_size
    ctx_list = [mx.gpu(x) for x in config.gpus]
    num_anchors = len(config.retinanet.network.SCALES) * len(config.retinanet.network.RATIOS)
    net = FCOSFPNNet(backbone, config.dataset.NUM_CLASSES, num_anchors)

    # Resume parameters.
    resume = None
    if resume is not None:
        params_coco = mx.nd.load(resume)
        for k in params_coco:
            params_coco[k.replace("arg:", "").replace("aux:", "")] = params_coco.pop(k)
        params = net.collect_params()

        for k in params.keys():
            try:
                params[k]._load_init(params_coco[k.replace('resnet0_', '')], ctx=mx.cpu())
                print("success load {}".format(k))
            except Exception as e:
                logging.exception(e)

    # Initialize parameters
    params = net.collect_params()
    for key in params.keys():
        if params[key]._data is None:
            default_init = mx.init.Zero() if "bias" in key or "offset" in key else mx.init.Xavier()
            default_init.set_verbosity(True)
            if params[key].init is not None and hasattr(params[key].init, "set_verbosity"):
                params[key].init.set_verbosity(True)
                params[key].initialize(init=params[key].init, default_init=params[key].init)
            else:
                params[key].initialize(default_init=default_init)
    if config.TRAIN.resume is not None:
        net.collect_params().load(config.TRAIN.resume)
        logging.info("loaded resume from {}".format(config.TRAIN.resume))

    net.collect_params().reset_ctx(list(set(ctx_list)))

    from data.bbox.mscoco import COCODetection
    if config.TRAIN.aspect_grouping:
        if config.dataset.dataset_type == "coco":
            from data.bbox.mscoco import COCODetection
            base_train_dataset = COCODetection(root=config.dataset.dataset_path, splits=("instances_train2017",),
                                               h_flip=config.TRAIN.FLIP, transform=None)
        elif config.dataset.dataset_type == "voc":
            from data.bbox.voc import VOCDetection
            base_train_dataset = VOCDetection(root=config.dataset.dataset_path,
                                              splits=((2007, 'trainval'), (2012, 'trainval')),
                                              preload_label=False)
        else:
            assert False
        train_dataset = AspectGroupingDataset(base_train_dataset, config,
                                              target_generator=RetinaNetTargetGenerator(config))
        train_loader = mx.gluon.data.DataLoader(dataset=train_dataset, batch_size=1, batchify_fn=batch_fn,
                                                num_workers=12, last_batch="discard", shuffle=True, thread_pool=False)
    else:
        assert False
    params_all = net.collect_params()
    params_to_train = {}
    params_fixed_prefix = config.network.FIXED_PARAMS
    for p in params_all.keys():
        ignore = False
        if params_fixed_prefix is not None:
            for f in params_fixed_prefix:
                if f in str(p) and "group" not in str(p):
                    ignore = True
                    params_all[p].grad_req = 'null'
                    logging.info("{} is ignored when training.".format(p))
        if not ignore: params_to_train[p] = params_all[p]
    lr_steps = [len(train_loader)  * int(x) for x in config.TRAIN.lr_step]
    logging.info(lr_steps)
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=lr_steps,
                                                        warmup_mode="constant", factor=.1,
                                                        base_lr=config.TRAIN.lr,
                                                        warmup_steps=config.TRAIN.warmup_step,
                                                        warmup_begin_lr=config.TRAIN.warmup_lr)

    trainer = mx.gluon.Trainer(
        params_to_train,  # fix batchnorm, fix first stage, etc...
        'sgd',
        {'wd': config.TRAIN.wd,
         'momentum': config.TRAIN.momentum,
         'clip_gradient': None,
         'lr_scheduler': lr_scheduler
         })
    # trainer = mx.gluon.Trainer(
    #     params_to_train,  # fix batchnorm, fix first stage, etc...
    #     'adam', {"learning_rate": 1e-4})
    # Please note that the GPU devices of the trainer states when saving must be same with that when loading.
    if config.TRAIN.trainer_resume is not None:
        trainer.load_states(config.TRAIN.trainer_resume)
        logging.info("loaded trainer states from {}.".format(config.TRAIN.trainer_resume))

    metric_loss_loc = mx.metric.Loss(name="loss_loc")
    metric_loss_cls = mx.metric.Loss(name="loss_cls")
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [metric_loss_loc, metric_loss_cls]:
        eval_metrics.add(child_metric)
    mobula.op.load("FocalLoss")
    for epoch in range(config.TRAIN.begin_epoch, config.TRAIN.end_epoch):
        net.hybridize(static_alloc=True, static_shape=False)
        for nbatch, data_batch in enumerate(tqdm.tqdm(train_loader, total=len(train_loader), unit_scale=1)):
            data_list = mx.gluon.utils.split_and_load(data_batch[0], ctx_list=ctx_list, batch_axis=0)
            targets_list = mx.gluon.utils.split_and_load(data_batch[1], ctx_list=ctx_list, batch_axis=0)
            losses = []
            losses_loc = []
            losses_cls = []
            with ag.record():
                for data, targets in zip(data_list, targets_list):
                    # targets: (2, 86, num_anchors, h x w)
                    fpn_predictions = net(data)
                    num_reg_channels = num_anchors * 4
                    cls_fpn_predictions = [x[:, num_reg_channels:].reshape(x.shape[0], -1, num_anchors, x.shape[2] * x.shape[3]) for x in fpn_predictions]
                    reg_fpn_predictions = [x[:, :num_reg_channels].reshape(x.shape[0], -1, num_anchors, x.shape[2] * x.shape[3]) for x in fpn_predictions]
                    cls_fpn_predictions = mx.nd.concat(*cls_fpn_predictions, dim=3)
                    reg_fpn_predictions = mx.nd.concat(*reg_fpn_predictions, dim=3)
                    mask_for_cls = targets[:, 0:1]
                    mask_for_reg = targets[:, 1:2]
                    num_pos = (mask_for_reg.sum() + 1)
                    loss_loc = mx.nd.smooth_l1(reg_fpn_predictions - targets[:, 2:6]) * mask_for_reg / 4 / num_pos
                    loss_cls = mobula.op.FocalLoss(alpha=.25, gamma=2, logits=cls_fpn_predictions, targets=targets[:, 6:]) * mask_for_cls / num_pos

                    losses.append(loss_loc)
                    losses.append(loss_cls)

                    losses_loc.append(loss_loc)
                    losses_cls.append(loss_cls)

            ag.backward(losses)
            trainer.step(len(ctx_list))
            for l in losses_loc:
                metric_loss_loc.update(None, l.sum())
            for l in losses_cls:
                metric_loss_cls.update(None, l.sum())
            if trainer.optimizer.num_update % config.TRAIN.log_interval == 0:
                msg = "Epoch={},Step={},lr={}, ".format(epoch, trainer.optimizer.num_update, trainer.learning_rate)
                msg += ','.join(['{}={:.3f}'.format(w, v) for w, v in zip(*eval_metrics.get())])
                logging.info(msg)
                eval_metrics.reset()

            if trainer.optimizer.num_update % 5000 == 0:
                save_path = os.path.join(config.TRAIN.log_path, "{}-{}.params".format(epoch, trainer.optimizer.num_update))
                net.collect_params().save(save_path)
                logging.info("Saved checkpoint to {}".format(save_path))
                trainer_path = save_path + "-trainer.states"
                trainer.save_states(trainer_path)
        save_path = os.path.join(config.TRAIN.log_path, "{}.params".format(epoch))
        net.collect_params().save(save_path)
        logging.info("Saved checkpoint to {}".format(save_path))
        trainer_path = save_path + "-trainer.states"
        trainer.save_states(trainer_path)


def parse_args():
    parser = argparse.ArgumentParser(description='QwQ')
    parser.add_argument('--dataset-type', help='voc or coco', required=False, type=str, default="coco")
    parser.add_argument('--num-classes', help='num-classes', required=False, type=int, default=81)
    parser.add_argument('--dataset-root', help='dataset root', required=False, type=str, default="/data1/coco")
    parser.add_argument('--gpus', help='The gpus used to train the network.', required=False, type=str, default="2,3")
    parser.add_argument('--demo', help='demo', action="store_true")

    args = parser.parse_args()
    return args


def main():
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    os.environ['MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF'] = '26'
    os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD'] = '999'
    os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD'] = '25'
    os.environ['MXNET_GPU_COPY_NTHREADS'] = '1'
    os.environ['MXNET_OPTIMIZER_AGGREGATION_SIZE'] = '54'
    # os.environ["MXNET_GPU_MEM_POOL_TYPE"] = "Round"
    args = parse_args()

    config = easydict.EasyDict()
    config.gpus = [int(x) for x in str(args.gpus).split(',')]
    config.dataset = easydict.EasyDict()
    config.dataset.NUM_CLASSES = args.num_classes
    config.dataset.dataset_type = args.dataset_type
    config.dataset.dataset_path = args.dataset_root
    config.retinanet = easydict.EasyDict()
    config.retinanet.network = easydict.EasyDict()
    config.retinanet.network.FPN_STRIDES = [8, 16, 32, 64, 128]
    config.retinanet.network.BASE_SIZES = [(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]
    config.retinanet.network.SCALES = [2**0, 2**(1/2), 2**(2/3)]
    config.retinanet.network.RATIOS = [1/2, 1, 2]
    config.retinanet.network.bbox_norm_coef = [0.1, 0.1, 0.2, 0.2]

    config.TRAIN = easydict.EasyDict()
    config.TRAIN.batch_size = 2 * len(config.gpus)
    config.TRAIN.lr = 0.01 * config.TRAIN.batch_size / 16
    config.TRAIN.warmup_lr = config.TRAIN.lr
    config.TRAIN.warmup_step = 1000
    config.TRAIN.wd = 1e-4
    config.TRAIN.momentum = .9
    config.TRAIN.log_path = "output/{}/RetinaNet-hflip".format(config.dataset.dataset_type, config.TRAIN.lr)
    config.TRAIN.log_interval = 100
    config.TRAIN.cls_focal_loss_alpha = .25
    config.TRAIN.cls_focal_loss_gamma = 2
    config.TRAIN.image_short_size = 600
    config.TRAIN.image_max_long_size = 1333
    config.TRAIN.aspect_grouping = True
    # if aspect_grouping is set to False, all images will be pad to (PAD_H, PAD_W)
    config.TRAIN.PAD_H = 768
    config.TRAIN.PAD_W = 768
    config.TRAIN.begin_epoch = 0
    config.TRAIN.end_epoch = 28
    config.TRAIN.lr_step = [12, 14]
    config.TRAIN.FLIP = True
    config.TRAIN.resume = None
    config.TRAIN.trainer_resume = None

    config.network = easydict.EasyDict()
    config.network.FIXED_PARAMS = []
    config.network.use_global_stats = True
    config.network.sync_bn = False

    os.makedirs(config.TRAIN.log_path, exist_ok=True)
    log_init(filename=os.path.join(config.TRAIN.log_path, "train_{}.log".format(time.time())))
    msg = pprint.pformat(config)
    logging.info(msg)
    if args.demo:
        demo_net(config)
    else:
        train_net(config)


def demo_net(config):
    import gluoncv
    backbone = FPNResNetV1(sync_bn=config.network.sync_bn, num_devices=len(config.gpus), use_global_stats=config.network.use_global_stats)
    # ctx_list = [mx.gpu(x) for x in config.gpus]
    ctx_list = [mx.cpu()]
    num_anchors = len(config.retinanet.network.SCALES) * len(config.retinanet.network.RATIOS)
    net = FCOSFPNNet(backbone, config.dataset.NUM_CLASSES, num_anchors)
    net.collect_params().load("output/coco/RetinaNet-hflip/0-5000.params")
    net.collect_params().reset_ctx(ctx_list[0])
    for x, y, z in os.walk("/data1/coco/val2017"):
        for name in z:
            path = os.path.join(x, name)
            image = cv2.imread(path)[:, :, ::-1]
            fscale = min(600 / min(image.shape[:2]), 1333 / max(image.shape[:2]))
            image_padded = cv2.resize(image, (0, 0), fx=fscale, fy=fscale)
            data = mx.nd.array(image_padded[np.newaxis], ctx=ctx_list[0])
            predictions = net(data)
            bboxes_pred_list = []
            num_anchors = len(config.retinanet.network.SCALES) * len(config.retinanet.network.RATIOS)
            predictions = [x.reshape(x.shape[0], -1, num_anchors, x.shape[2], x.shape[3]) for x in predictions]
            for fpn_prediction, base_size in zip(predictions, config.retinanet.network.BASE_SIZES):
                stride = image_padded.shape[0] // fpn_prediction.shape[2]
                fpn_prediction[:, 4:] = fpn_prediction[:, 4:].sigmoid()
                fpn_prediction_reshaped_np = fpn_prediction.transpose((0, 3, 4, 2, 1)).asnumpy()
                # fpn_prediction_reshaped_np[:, :, :, :, :4] *= np.array(config.retinanet.network.bbox_norm_coef)[None, None, None, None]
                rois = mobula.op.RetinaNetRegression[np.ndarray](number_of_classes=config.dataset.NUM_CLASSES,
                                                                 base_size=base_size,
                                                                 cls_threshold=.01,
                                                                 stride = stride
                                                                 )(image=data.asnumpy(),
                                                                   feature=fpn_prediction_reshaped_np)
                # plt.imshow(fpn_prediction_reshaped_np[0, :, :, :, 4:].max(axis=2).max(axis=2))
                # plt.show()
                rois = rois[0]
                rois = rois[np.argsort(-1 * rois[:, 4])[:200]]
                bboxes_pred_list.append(rois)
            bboxes_pred = np.concatenate(bboxes_pred_list, axis=0)
            cls_dets = mx.nd.contrib.box_nms(mx.nd.array(bboxes_pred, ctx=mx.cpu()),
                                          overlap_thresh=.3, coord_start=0, score_index=4, id_index=-1,
                                          force_suppress=True, in_format='corner',
                                          out_format='corner').asnumpy()
            cls_dets = cls_dets[np.argsort(-1 * cls_dets[:, 4])[:10]]
            # cls_dets = bboxes_pred

            gluoncv.utils.viz.plot_bbox(image_padded, bboxes=cls_dets[:, :4], scores=cls_dets[:, 4], labels=cls_dets[:, 5],
                                        thresh=0.01, class_names=gluoncv.data.COCODetection.CLASSES)
            plt.show()
            pass


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
