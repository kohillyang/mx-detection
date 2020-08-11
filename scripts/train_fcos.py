from __future__ import print_function

import logging
import os
import pprint
import sys

import cv2
import mxnet as mx
import mxnet.autograd as ag
import numpy as np
import tqdm
import time
import easydict
import matplotlib
import gluoncv
import matplotlib.pyplot as plt
from models.fcos.resnet import ResNet
from models.fcos.resnetv1b import FPNResNetV1
from utils.common import log_init
from data.bbox.bbox_dataset import AspectGroupingDataset
from utils.lrsheduler import WarmupMultiFactorScheduler

sys.path.append(os.path.join(os.path.dirname(__file__), "../MobulaOP"))
import data.transforms.bbox as bbox_t
import mobula
setattr(mobula.config, "NVCC", "/usr/local/cuda-10.0/bin/nvcc")
mobula.op.load('FCOSTargetGenerator', os.path.join(os.path.dirname(__file__), "../utils/operator_cxx"))
mobula.op.load('FCOSRegression', os.path.join(os.path.dirname(__file__), "../utils/operator_cxx"))
import argparse

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


def BCEFocalLoss(x, target):
    alpha = .25
    p = x.sigmoid()
    loss = alpha * target * ((1-p)**2) * mx.nd.log(p + 1e-11)
    loss = loss + (1-alpha) * (1-target) * (p **2) * mx.nd.log(1 - p + 1e-11)
    return -loss


@mobula.op.register
class L2Loss:
    def forward(self, y, target):
        # return 2 * mx.nd.log(1 + mx.nd.exp(y)) - y - target * y
        return (y - target) ** 2

    def backward(self, dy):
        # grad = mx.nd.sigmoid(self.X[0])*2 - 1 - self.X[1]
        grad = 2 * (self.X[0] - self.X[1])
        # grad *= 1e-4
        self.dX[0][:] = grad * dy

    def infer_shape(self, in_shape):
        assert in_shape[0] == in_shape[1]
        return in_shape, [in_shape[0]]


class IoULoss(mx.gluon.nn.Block):
    def __init__(self):
        super(IoULoss, self).__init__()

    def max(self, *args):
        if len(args) == 1:
            return args[0]
        else:
            maximum = args[0]
            for arg in args[1:]:
                maximum = mx.nd.maximum(maximum, arg)
            return maximum

    def forward(self, prediction, target):
        assert prediction.shape[1] == 4
        assert target.shape[1] == 4
        target = mx.nd.maximum(target, mx.nd.ones_like(target))
        target = mx.nd.log(target)

        l, t, r, b = 0, 1, 2, 3 # l, t, r, b
        tl = target[:, t] + target[:, l]
        tr = target[:, t] + target[:, r]
        bl = target[:, b] + target[:, l]
        br = target[:, b] + target[:, r]
        tl_hat = prediction[:, t] + prediction[:, l]
        tr_hat = prediction[:, t] + prediction[:, r]
        bl_hat = prediction[:, b] + prediction[:, l]
        br_hat = prediction[:, b] + prediction[:, r]

        x_t_i = mx.nd.minimum(target[:, t], prediction[:, t])
        x_b_i = mx.nd.minimum(target[:, b], prediction[:, b])
        x_l_i = mx.nd.minimum(target[:, l], prediction[:, l])
        x_r_i = mx.nd.minimum(target[:, r], prediction[:, r])

        tl_i = x_t_i + x_l_i
        tr_i = x_t_i + x_r_i
        bl_i = x_b_i + x_l_i
        br_i = x_b_i + x_r_i

        max_v = self.max(tl, tr, bl, br, tl_hat, tr_hat, bl_hat, br_hat, tl_i, tr_i, bl_i, br_i)
        I = mx.nd.exp(tl_i - max_v) + mx.nd.exp(tr_i- max_v) + mx.nd.exp(bl_i- max_v) + mx.nd.exp(br_i- max_v)
        X = mx.nd.exp(tl- max_v) + mx.nd.exp(tr- max_v) + mx.nd.exp(bl- max_v) + mx.nd.exp(br- max_v)
        X_hat = mx.nd.exp(tl_hat- max_v) + mx.nd.exp(tr_hat- max_v) + mx.nd.exp(bl_hat- max_v) + mx.nd.exp(br_hat- max_v)
        I_over_U = I / (X + X_hat - I) + 1e-7
        return -I_over_U.log()


def batch_fn(x):
    return x


class FCOS_Head(mx.gluon.nn.HybridBlock):
    def __init__(self, num_classes):
        super(FCOS_Head, self).__init__()
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
            self.feat_cls.add(mx.gluon.nn.Conv2D(channels=num_classes-1, kernel_size=1, padding=0,
                                                 bias_initializer=init_bias, weight_initializer=init))

            self.feat_reg = mx.gluon.nn.HybridSequential()
            for i in range(4):
                self.feat_reg.add(mx.gluon.nn.Conv2D(channels=256, kernel_size=3, padding=1, weight_initializer=init))
                self.feat_reg.add(mx.gluon.nn.GroupNorm(num_groups=32))
                self.feat_reg.add(mx.gluon.nn.Activation(activation="relu"))

            # one extra channel for center-ness, four channel for location regression.
            self.feat_reg_loc = mx.gluon.nn.Conv2D(channels=4, kernel_size=1, padding=0, weight_initializer=init)
            self.feat_reg_centerness = mx.gluon.nn.Conv2D(channels=1, kernel_size=1, padding=0, weight_initializer=init)

    def hybrid_forward(self, F, x, scale):
        feat_reg = self.feat_reg(x)
        x_loc = F.broadcast_mul(self.feat_reg_loc(feat_reg), scale)
        x_centerness = self.feat_reg_centerness(feat_reg)
        x_cls = self.feat_cls(x)
        x = F.concat(x_loc, x_centerness, x_cls, dim=1)
        return x


class FCOSFPNNet(mx.gluon.nn.HybridBlock):
    def __init__(self, backbone, num_classes):
        super(FCOSFPNNet, self).__init__()
        self.backbone = backbone
        self.fcos_head = FCOS_Head(num_classes)
        with self.name_scope():
            self.scale0 = self.params.get('scale0', shape=[1, 1, 1, 1],
                                          init=mx.init.One(),  # mx.nd.array(),
                                          allow_deferred_init=False, grad_req='write')
            self.scale1 = self.params.get('scale1', shape=[1, 1, 1, 1],
                                          init=mx.init.One(),  # mx.nd.array(),
                                          allow_deferred_init=False, grad_req='write')
            self.scale2 = self.params.get('scale2', shape=[1, 1, 1, 1],
                                          init=mx.init.One(),  # mx.nd.array(),
                                          allow_deferred_init=False, grad_req='write')
            self.scale3 = self.params.get('scale3', shape=[1, 1, 1, 1],
                                          init=mx.init.One(),  # mx.nd.array(),
                                          allow_deferred_init=False, grad_req='write')
            self.scale4 = self.params.get('scale4', shape=[1, 1, 1, 1],
                                          init=mx.init.One(),  # mx.nd.array(),
                                          allow_deferred_init=False, grad_req='write')

    def hybrid_forward(self, F, x, scale0, scale1, scale2, scale3, scale4):
        # typically the strides are (4, 8, 16, 32, 64)
        scales = [scale0, scale1, scale2, scale3, scale4]
        x = self.backbone(x)
        if isinstance(x, list) or isinstance(x, tuple):
            return [self.fcos_head(xx, s) for xx, s in zip(x, scales)]
        else:
            return [self.fcos_head(x)]


def batch_fn(x):
    return x


class FCOSTargetGenerator(object):
    def __init__(self, config):
        super(FCOSTargetGenerator, self).__init__()
        self.config = config
        self.strides = config.FCOS.network.FPN_SCALES
        self.fpn_min_distance = config.FCOS.network.FPN_MINIMUM_DISTANCES
        self.fpn_max_distance = config.FCOS.network.FPN_MAXIMUM_DISTANCES
        self.number_of_classes = config.dataset.NUM_CLASSES

    def __call__(self, image_transposed, bboxes):
        h, w, c = image_transposed.shape
        bboxes = bboxes.copy()
        bboxes[:, 4] += 1
        outputs = [image_transposed]
        # fig, axes = plt.subplots(3, 3)
        # axes = axes.reshape(-1)
        # n_axes = 0
        num_pos = 0
        for stride, min_distance, max_distance in zip(self.strides, self.fpn_min_distance, self.fpn_max_distance):
            target = mobula.op.FCOSTargetGenerator[np.ndarray](stride, min_distance, max_distance, self.number_of_classes)(
                image_transposed.astype(np.float32), bboxes.astype(np.float32))

            num_pos += target[:, :, 0].sum()
            # fig, axes = plt.subplots(1, 3)
            # axes[0].imshow(target[:, :, 1:5].max(axis=2))
            # axes[1].imshow(target[:, :, 0])
            # axes[2].imshow(target[:, :, 5])
            # plt.show()
            target = target.transpose((2, 0, 1))
            # axes[n_axes].imshow(target[6:].max(axis=0))
            # n_axes += 1
            outputs.append(target)
        num_pos = max(num_pos, 1)
        outputs.append(np.array([num_pos])[np.newaxis, np.newaxis])
        # axes[n_axes].imshow(image_transposed.astype(np.uint8))
        # gluoncv.utils.viz.plot_bbox(image_transposed, bboxes=bboxes[:, :4], ax=axes[n_axes])
        # plt.show()
        outputs = tuple(mx.nd.array(x) for x in outputs)
        if self.config.TRAIN.USE_FP16:
            outputs = tuple(x.astype("float16") for x in outputs)
        return outputs


def train_net(config):
    mx.random.seed(3)
    np.random.seed(3)

    backbone = FPNResNetV1(sync_bn=config.network.sync_bn, num_devices=len(config.gpus),
                           use_global_stats=config.network.use_global_stats)
    batch_size = config.TRAIN.batch_size
    ctx_list = [mx.gpu(x) for x in config.gpus]
    net = FCOSFPNNet(backbone, config.dataset.NUM_CLASSES)
    if config.TRAIN.USE_FP16:
        net.cast("float16")
        from mxnet.contrib import amp
        amp.init()
    if config.use_hvd:
        import horovod.mxnet as hvd
        hvd.init()

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
            default_init = mx.init.Zero() if "bias" in key or "offset" in key else mx.init.Normal()
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
        train_dataset = AspectGroupingDataset(base_train_dataset, config, target_generator=FCOSTargetGenerator(config))
        train_loader = mx.gluon.data.DataLoader(dataset=train_dataset, batch_size=1, batchify_fn=batch_fn,
                                                num_workers=16, last_batch="discard", shuffle=True, thread_pool=False)
    else:
        train_transforms = bbox_t.Compose([
            # Flipping is implemented in dataset.
            bbox_t.ResizePad(dst_h=config.TRAIN.PAD_H, dst_w=config.TRAIN.PAD_W),
            FCOSTargetGenerator(config)
        ])
        from data.bbox.mscoco import COCODetection
        train_dataset = COCODetection(root=config.dataset.dataset_path, splits=("instances_train2017",),
                                      h_flip=config.TRAIN.FLIP, transform=train_transforms)
        train_loader = mx.gluon.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                num_workers=16, last_batch="discard", shuffle=True, thread_pool=False)

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
    lr_steps = [len(train_loader) * int(x) for x in config.TRAIN.lr_step]
    logging.info(lr_steps)
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=lr_steps,
                                                        warmup_mode="constant", factor=.1,
                                                        base_lr=config.TRAIN.lr,
                                                        warmup_steps=config.TRAIN.warmup_step,
                                                        warmup_begin_lr=config.TRAIN.warmup_lr)
    if config.use_hvd:
        hvd.broadcast_parameters(net.collect_params(), root_rank=0)
        trainer = hvd.DistributedTrainer(
            params_to_train,
            'sgd',
            {'wd': config.TRAIN.wd,
             'momentum': config.TRAIN.momentum,
             'clip_gradient': None,
             'lr_scheduler': lr_scheduler,
             'multi_precision': True,
             })
    else:
        trainer = mx.gluon.Trainer(
            params_to_train,  # fix batchnorm, fix first stage, etc...
            'sgd',
            {'wd': config.TRAIN.wd,
             'momentum': config.TRAIN.momentum,
             'clip_gradient': None,
             'lr_scheduler': lr_scheduler,
             'multi_precision': True,
             })
        if config.TRAIN.USE_FP16:
            amp.init_trainer(trainer)
    # trainer = mx.gluon.Trainer(
    #     params_to_train,  # fix batchnorm, fix first stage, etc...
    #     'adam', {"learning_rate": 4e-4})
    # Please note that the GPU devices of the trainer states when saving must be same with that when loading.
    if config.TRAIN.trainer_resume is not None:
        trainer.load_states(config.TRAIN.trainer_resume)
        logging.info("loaded trainer states from {}.".format(config.TRAIN.trainer_resume))

    metric_loss_loc = mx.metric.Loss(name="loss_loc")
    metric_loss_cls = mx.metric.Loss(name="loss_cls")
    metric_loss_center = mx.metric.Loss(name="loss_center")
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [metric_loss_loc, metric_loss_cls, metric_loss_center]:
        eval_metrics.add(child_metric)

    for epoch in range(config.TRAIN.begin_epoch, config.TRAIN.end_epoch):
        net.hybridize(static_alloc=True, static_shape=False)
        for nbatch, data_batch in enumerate(tqdm.tqdm(train_loader, total=len(train_loader), unit_scale=1)):
            data_list = mx.gluon.utils.split_and_load(data_batch[0], ctx_list=ctx_list, batch_axis=0)
            label_0_list = mx.gluon.utils.split_and_load(data_batch[1], ctx_list=ctx_list, batch_axis=0)
            label_1_list = mx.gluon.utils.split_and_load(data_batch[2], ctx_list=ctx_list, batch_axis=0)
            label_2_list = mx.gluon.utils.split_and_load(data_batch[3], ctx_list=ctx_list, batch_axis=0)
            label_3_list = mx.gluon.utils.split_and_load(data_batch[4], ctx_list=ctx_list, batch_axis=0)
            label_4_list = mx.gluon.utils.split_and_load(data_batch[5], ctx_list=ctx_list, batch_axis=0)
            number_of_positive_list = mx.gluon.utils.split_and_load(data_batch[6], ctx_list=ctx_list, batch_axis=0)

            losses = []
            losses_loc = []
            losses_center_ness = []
            losses_cls=[]
            with ag.record():
                for data, label0, label1, label2, label3, label4, no_pos in zip(data_list, label_0_list,
                                                                        label_1_list, label_2_list, label_3_list,
                                                                        label_4_list, number_of_positive_list
                                                                        ):
                    labels = [label0, label1, label2, label3, label4]
                    labels = mx.nd.concat(*[x.reshape((0, 0, -1)) for x in labels], dim=2)
                    fpn_predictions = net(data)
                    preds = mx.nd.concat(*[x.reshape((0, 0, -1)) for x in fpn_predictions], dim=2)
                    num_pos = labels[:, 6:].max(axis=1).sum()

                    reg_mask = labels[:, 0]
                    iou_loss = mx.nd.where(reg_mask, IoULoss()(preds[:, :4], labels[:, 1:5]),
                                            mx.nd.zeros_like(reg_mask)) * labels[:, 5] / (labels[:, 5].sum() + 1)
                    loss_center = BCELoss(preds[:, 4], labels[:, 5]) * labels[:, 0] / num_pos
                    loss_cls = BCEFocalLoss(preds[:, 5:], labels[:, 6:]) / num_pos
                    losses.append(iou_loss)
                    losses.append(loss_cls)
                    losses.append(loss_center)
                    losses_loc.append(iou_loss)
                    losses_center_ness.append(loss_center)
                    losses_cls.append(loss_cls)
            ag.backward(losses)
            trainer.step(batch_size)
            for l in losses_loc:
                metric_loss_loc.update(None, l.sum())
            for l in losses_center_ness:
                metric_loss_center.update(None, l.sum())
            for l in losses_cls:
                metric_loss_cls.update(None, l.sum())
            if trainer.optimizer.num_update % config.TRAIN.log_interval == 0:
                msg = "Epoch={},Step={},lr={}, ".format(epoch, trainer.optimizer.num_update, trainer.learning_rate)
                msg += ','.join(['{}={:.3f}'.format(w, v) for w, v in zip(*eval_metrics.get())])
                logging.info(msg)
                eval_metrics.reset()

                # plt.imshow(data[0].asnumpy().astype(np.uint8))
                # plt.savefig(os.path.join(config.TRAIN.log_path, "{}_image.jpg".format(trainer.optimizer.num_update)))
                #
                # plt.imshow(class_prediction[0].sigmoid().max(axis=0).asnumpy().astype(np.float32))
                # plt.savefig(os.path.join(config.TRAIN.log_path, "{}_heatmap.jpg".format(trainer.optimizer.num_update)))
                # plt.imshow(class_target[0].max(axis=0).asnumpy().astype(np.float32))
                # plt.savefig(os.path.join(config.TRAIN.log_path, "{}_heatmap_target.jpg".format(trainer.optimizer.num_update)))
                #
                # plt.imshow(loc_prediction[0, 0].asnumpy().astype(np.float32))
                # plt.savefig(os.path.join(config.TRAIN.log_path, "{}_bexp.jpg".format(trainer.optimizer.num_update)))
                #
                # plt.imshow(loc_prediction[0, 0].exp().asnumpy().astype(np.float32))
                # plt.savefig(os.path.join(config.TRAIN.log_path, "{}_exp.jpg".format(trainer.optimizer.num_update)))
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
    os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    os.environ['MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF'] = '26'
    os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD'] = '999'
    os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD'] = '25'
    os.environ['MXNET_GPU_COPY_NTHREADS'] = '1'
    os.environ['MXNET_OPTIMIZER_AGGREGATION_SIZE'] = '54'
    # os.environ["MXNET_GPU_MEM_POOL_TYPE"] = "Round"
    args = parse_args()

    config = easydict.EasyDict()
    config.gpus = [int(x) for x in str(args.gpus).split(',')]
    config.use_hvd = False

    config.dataset = easydict.EasyDict()
    config.dataset.NUM_CLASSES = args.num_classes
    config.dataset.dataset_type = args.dataset_type
    config.dataset.dataset_path = args.dataset_root

    config.FCOS = easydict.EasyDict()
    config.FCOS.network = easydict.EasyDict()

    config.FCOS.network.FPN_SCALES = [8, 16, 32, 64, 128]
    config.FCOS.network.FPN_MINIMUM_DISTANCES = [0, 64, 128, 256, 512]
    config.FCOS.network.FPN_MAXIMUM_DISTANCES = [64, 128, 256, 512, 4096]
    config.TRAIN = easydict.EasyDict()
    config.TRAIN.batch_size = 2 * len(config.gpus)
    config.TRAIN.lr = 0.01 * config.TRAIN.batch_size / 16
    config.TRAIN.warmup_lr = config.TRAIN.lr
    config.TRAIN.warmup_step = 1000
    config.TRAIN.wd = 1e-4
    config.TRAIN.momentum = .9
    config.TRAIN.log_path = "output/{}/focal_alpha_gamma_lr_{}".format(config.dataset.dataset_type, config.TRAIN.lr)
    config.TRAIN.log_interval = 20
    config.TRAIN.cls_focal_loss_alpha = .25
    config.TRAIN.cls_focal_loss_gamma = 2
    config.TRAIN.image_short_size = 800
    config.TRAIN.image_max_long_size = 1333
    config.TRAIN.aspect_grouping = True
    # if aspect_grouping is set to False, all images will be pad to (PAD_H, PAD_W)
    config.TRAIN.PAD_H = 768
    config.TRAIN.PAD_W = 768
    config.TRAIN.begin_epoch = 0
    config.TRAIN.end_epoch = 28
    config.TRAIN.lr_step = [16, 20]
    config.TRAIN.FLIP = False
    config.TRAIN.resume = None
    config.TRAIN.trainer_resume = None
    config.TRAIN.USE_FP16 = False
    if config.TRAIN.USE_FP16:
        os.environ["MXNET_SAFE_ACCUMULATION"] = "1"
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


def inference_one_image(net, ctx, image_path):
    image = cv2.imread(image_path)[:, :, ::-1]
    fscale = min(600 / min(image.shape[:2]), 1333 / max(image.shape[:2]))
    image_padded = cv2.resize(image, (0, 0), fx=fscale, fy=fscale)
    predictions = net(mx.nd.array(image_padded[np.newaxis], ctx=ctx))
    bboxes_pred_list = []
    for pred in predictions:
        stride = image_padded.shape[0] // pred.shape[2]

        pred[:, :4] = (pred[:, :4]).exp()
        pred[:, 4] = pred[:, 4].sigmoid()
        pred[:, 5:] = pred[:, 5:].sigmoid()

        pred_np = pred.asnumpy()
        rois = mobula.op.FCOSRegression[np.ndarray](stride)(prediction=pred_np)[0]
        rois = rois[np.argsort(-1 * rois[:, 4])[:200]]
        bboxes_pred_list.append(rois)
    bboxes_pred = np.concatenate(bboxes_pred_list, axis=0)
    if len(bboxes_pred > 0):
        cls_dets = mx.nd.contrib.box_nms(mx.nd.array(bboxes_pred, ctx=mx.cpu()),
                                         overlap_thresh=.5, coord_start=0, score_index=4, id_index=-1,
                                         force_suppress=True, in_format='corner',
                                         out_format='corner').asnumpy()
        cls_dets = cls_dets[np.where(cls_dets[:, 4] > 0.01)]
        cls_dets[:, :4] /= fscale
        # gluoncv.utils.viz.plot_bbox(image, bboxes=cls_dets[:, :4], scores=cls_dets[:, 4], labels=cls_dets[:, 5],
        #                             thresh=0.2, class_names=gluoncv.data.COCODetection.CLASSES)
        # plt.show()
        cls_dets[:, 5] += 1
        return cls_dets
    else:
        return []


def demo_net(config):
    import json
    from utils.evaluate import evaluate_coco
    import tqdm
    ctx_list = [mx.gpu(3)]
    backbone = FPNResNetV1(sync_bn=True, num_devices=3, use_global_stats=True)
    net = FCOSFPNNet(backbone, config.dataset.NUM_CLASSES)
    net.collect_params().load("/data1/kohill/mx-detection/output/focal_alpha_gamma_lr_0.00375/11-610000.params")
    net.collect_params().reset_ctx(ctx_list[0])
    results = {}
    results["results"] = []
    for x, y, names in os.walk("/data1/coco/val2017"):
        for name in tqdm.tqdm(names[:100]):
            one_img = {}
            one_img["filename"] = os.path.basename(name)
            one_img["rects"] = []
            preds = inference_one_image(net, ctx_list[0], os.path.join(x, name))
            for i in range(len(preds)):
                one_rect = {}
                xmin, ymin, xmax, ymax = preds[i][:4]
                one_rect["xmin"] = int(np.round(xmin))
                one_rect["ymin"] = int(np.round(ymin))
                one_rect["xmax"] = int(np.round(xmax))
                one_rect["ymax"] = int(np.round(ymax))
                one_rect["confidence"] = float(preds[i][4])
                one_rect["label"] = int(preds[i][5])
                one_img["rects"].append(one_rect)
            results["results"].append(one_img)
    save_path = 'results.json'
    json.dump(results, open(save_path, "wt"))
    evaluate_coco(json_label="/data1/coco/annotations/instances_val2017.json", json_predict=save_path)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
