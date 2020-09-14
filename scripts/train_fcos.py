import argparse
import logging
import os
import pprint
import time

import cv2
import easydict
import gluoncv
import matplotlib.pyplot as plt
import mobula
import mxnet as mx
import mxnet.autograd as ag
import numpy as np
import tqdm

from data.bbox.bbox_dataset import AspectGroupingDataset
from utils.common import log_init
from models.backbones.resnet import ResNetV1B
# from models.backbones.dcn_resnet.resnet import ResNetV1B


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

    def forward(self, prediction_bbox, target_bbox):
        assert prediction_bbox.shape[1] == 4
        assert target_bbox.shape[1] == 4
        prediction_bbox = mx.nd.clip(prediction_bbox, 0, 4096)

        l, t, r, b = 0, 1, 2, 3 # l, t, r, b
        # tl = target_bbox[:, t] + target_bbox[:, l]
        # tr = target_bbox[:, t] + target_bbox[:, r]
        # bl = target_bbox[:, b] + target_bbox[:, l]
        # br = target_bbox[:, b] + target_bbox[:, r]
        lr = target_bbox[:, l] + target_bbox[:, r]
        tb = target_bbox[:, t] + target_bbox[:, b]

        lr_hat = prediction_bbox[:, l] + prediction_bbox[:, r]
        tb_hat = prediction_bbox[:, t] + prediction_bbox[:, b]

        x_t_i = mx.nd.minimum(target_bbox[:, t], prediction_bbox[:, t])
        x_b_i = mx.nd.minimum(target_bbox[:, b], prediction_bbox[:, b])
        x_l_i = mx.nd.minimum(target_bbox[:, l], prediction_bbox[:, l])
        x_r_i = mx.nd.minimum(target_bbox[:, r], prediction_bbox[:, r])

        I = (x_l_i + x_r_i) * (x_b_i + x_t_i)
        X = lr * tb
        X_hat = lr_hat * tb_hat
        I_over_U = (I + 1) / (X + X_hat - I + 1)
        return -(I_over_U).log()


def load_mobula_ops():
    logging.info(mobula.__path__)
    mobula.op.load('BCELoss', os.path.join(os.path.dirname(__file__), "../utils/operator_cxx"))
    mobula.op.load('FCOSTargetGenerator', os.path.join(os.path.dirname(__file__), "../utils/operator_cxx"))
    mobula.op.load('FCOSRegression', os.path.join(os.path.dirname(__file__), "../utils/operator_cxx"))


def batch_fn(x):
    return x


class PyramidNeckFCOS(mx.gluon.nn.HybridBlock):
    def __init__(self, feature_dim=256):
        super(PyramidNeckFCOS, self).__init__()
        self.fpn_p7_3x3 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=3, prefix="fpn_p7_1x1_", strides=2, padding=1)
        self.fpn_p6_3x3 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=3, prefix="fpn_p6_1x1_", strides=2, padding=1)
        self.fpn_p5_1x1 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=1, prefix="fpn_p5_1x1_")
        self.fpn_p4_1x1 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=1, prefix="fpn_p4_1x1_")
        self.fpn_p3_1x1 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=1, prefix="fpn_p3_1x1_")

    def hybrid_forward(self, F, res2, res3, res4, res5):
        fpn_p5_1x1 = self.fpn_p5_1x1(res5)
        fpn_p4_1x1 = self.fpn_p4_1x1(res4)
        fpn_p3_1x1 = self.fpn_p3_1x1(res3)

        fpn_p5_upsample = F.contrib.BilinearResize2D(fpn_p5_1x1, mode="like", like=fpn_p4_1x1)
        fpn_p4_plus = F.ElementWiseSum(*[fpn_p5_upsample, fpn_p4_1x1])
        fpn_p4_upsample = F.contrib.BilinearResize2D(fpn_p4_plus, mode="like", like=fpn_p3_1x1)
        fpn_p3_plus = F.ElementWiseSum(*[fpn_p4_upsample, fpn_p3_1x1])

        p6 = self.fpn_p6_3x3(res5)
        p7 = self.fpn_p7_3x3(F.relu(p6))

        return fpn_p3_plus, fpn_p4_plus, fpn_p5_1x1, p6, p7


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
        x = F.concat(x_loc, x_centerness, dim=1), x_cls
        return x


class FCOSFPNNet(mx.gluon.nn.HybridBlock):
    def __init__(self, backbone, num_classes):
        super(FCOSFPNNet, self).__init__()
        self.backbone = backbone
        self.fcos_head = FCOS_Head(num_classes)
        with self.name_scope():
            self.scale0 = self.params.get('scale0', shape=[1, 1, 1, 1],
                                          init=mx.init.Constant(1),  # mx.nd.array(),
                                          allow_deferred_init=False, grad_req='write', wd_mult=0)
            self.scale1 = self.params.get('scale1', shape=[1, 1, 1, 1],
                                          init=mx.init.Constant(1),  # mx.nd.array(),
                                          allow_deferred_init=False, grad_req='write', wd_mult=0)
            self.scale2 = self.params.get('scale2', shape=[1, 1, 1, 1],
                                          init=mx.init.Constant(1),  # mx.nd.array(),
                                          allow_deferred_init=False, grad_req='write', wd_mult=0)
            self.scale3 = self.params.get('scale3', shape=[1, 1, 1, 1],
                                          init=mx.init.Constant(1),  # mx.nd.array(),
                                          allow_deferred_init=False, grad_req='write', wd_mult=0)
            self.scale4 = self.params.get('scale4', shape=[1, 1, 1, 1],
                                          init=mx.init.Constant(1),  # mx.nd.array(),
                                          allow_deferred_init=False, grad_req='write', wd_mult=0)

    def hybrid_forward(self, F, x, scale0, scale1, scale2, scale3, scale4):
        # typically the strides are (4, 8, 16, 32, 64)
        scales = [scale0, scale1, scale2, scale3, scale4]
        x = self.backbone(x)
        if isinstance(x, list) or isinstance(x, tuple):
            outputs = [self.fcos_head(xx, s) for xx, s in zip(x, scales)]
        else:
            outputs = [self.fcos_head(xx, s) for xx, s in zip(x, scales)]
        loc_outputs = [x[0] for x in outputs]
        loc_outputs = [x.reshape((0, 0, -1)) for x in loc_outputs]
        loc_outputs = F.concat(*loc_outputs, dim=2)

        cls_outputs = [x[1] for x in outputs]
        cls_outputs = [x.reshape((0, 0, -1)) for x in cls_outputs]
        cls_outputs = F.concat(*cls_outputs, dim=2)

        return loc_outputs, cls_outputs


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
        targets = []
        for stride, min_distance, max_distance in zip(self.strides, self.fpn_min_distance, self.fpn_max_distance):
            target = mobula.op.FCOSTargetGenerator[np.ndarray](stride, min_distance, max_distance, self.number_of_classes)(
                image_transposed.astype(np.float32), bboxes.astype(np.float32))

            target = target.transpose((2, 0, 1))
            target = target.reshape((target.shape[0], -1))
            targets.append(target)
        targets = np.concatenate(targets, axis=1)
        outputs.append(targets)
        outputs = tuple(np.array(x) for x in outputs)
        return outputs


def train_net(config):
    mx.random.seed(3)
    np.random.seed(3)

    if config.TRAIN.USE_FP16:
        from mxnet.contrib import amp
        amp.init()
    if config.use_hvd:
        import horovod.mxnet as hvd

    ctx_list = [mx.gpu(x) for x in config.gpus]
    neck = PyramidNeckFCOS(feature_dim=config.network.fpn_neck_feature_dim)
    backbone = ResNetV1B(neck=neck,
                         sync_bn=config.network.sync_bn, num_devices=len(config.gpus),
                         use_global_stats=config.network.use_global_stats)
    net = FCOSFPNNet(backbone, config.dataset.NUM_CLASSES)

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

    if config.use_hvd:
        class SplitDataset(object):
            def __init__(self, da, local_size, local_rank):
                self.da = da
                self.local_size = local_size
                self.locak_rank = local_rank

            def __len__(self):
                return len(self.da) // self.local_size

            def __getitem__(self, idx):
                return self.da[idx * self.local_size + self.locak_rank]

        train_dataset = SplitDataset(train_dataset, local_size=hvd.local_size(), local_rank=hvd.local_rank())

    train_loader = mx.gluon.data.DataLoader(dataset=train_dataset, batch_size=1,
                                            num_workers=8, last_batch="discard", shuffle=True,
                                            thread_pool=False, batchify_fn=batch_fn)

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
             },
            update_on_kvstore=(False if config.TRAIN.USE_FP16 else None), kvstore=mx.kvstore.create('local')
        )
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
    mobula.op.load("FocalLoss")
    mobula.op.load("IoULoss", os.path.join(os.path.dirname(__file__), "../utils/operator_cxx"))

    net.hybridize(static_alloc=True, static_shape=False)
    for ctx in ctx_list:
        with ag.record():
            _ = net(mx.nd.random.randn(1, config.TRAIN.image_max_long_size, config.TRAIN.image_short_size, 3, ctx=ctx))
        ag.backward(_)
        del _
        net.collect_params().zero_grad()
    mx.nd.waitall()

    while trainer.optimizer.num_update <= config.TRAIN.end_epoch * len(train_loader):
        epoch = trainer.optimizer.num_update // len(train_loader)
        for data_batch in tqdm.tqdm(train_loader) if not config.use_hvd or hvd.local_rank() == 0 else train_loader:
            if config.use_hvd:
                data_list = [data_batch[0].as_in_context(ctx_list[0])]
                targets_list = [data_batch[1].as_in_context(ctx_list[0])]
            else:
                if isinstance(data_batch[0], mx.nd.NDArray):
                    data_list = mx.gluon.utils.split_and_load(mx.nd.array(data_batch[0]), ctx_list=ctx_list, batch_axis=0)
                    targets_list = mx.gluon.utils.split_and_load(mx.nd.array(data_batch[1]), ctx_list=ctx_list, batch_axis=0)
                else:
                    data_list = mx.gluon.utils.split_and_load(mx.nd.array(data_batch[0][0]), ctx_list=ctx_list, batch_axis=0)
                    targets_list = mx.gluon.utils.split_and_load(mx.nd.array(data_batch[0][1]), ctx_list=ctx_list, batch_axis=0)

            losses_loc = []
            losses_center_ness = []
            losses_cls=[]
            with ag.record():
                for data, targets in zip(data_list, targets_list):
                    loc_preds, cls_preds = net(data)
                    num_pos = targets[:, 0].sum()
                    num_pos_denominator = mx.nd.maximum(num_pos, mx.nd.ones_like(num_pos))
                    centerness_sum = targets[:, 5].sum()
                    centerness_sum_denominator = mx.nd.maximum(centerness_sum, mx.nd.ones_like(centerness_sum))
                    loc_preds_transposed_reshaped = loc_preds[:, :4].transpose((0, 2, 1)).reshape((-1, 4))
                    loc_targets_transposed_reshaped = targets[:, 1:5].transpose((0, 2, 1)).reshape((-1, 4))
                    loc_centerness_mask = targets[:, 5:6].transpose((0, 2, 1)).reshape((-1, 1))
                    iou_loss = mobula.op.IoULoss(loc_preds_transposed_reshaped, loc_targets_transposed_reshaped)
                    iou_loss = iou_loss * loc_centerness_mask / centerness_sum_denominator
                    # iou_loss = IoULoss()(loc_preds[:, :4], targets[:, 1:5]) * targets[:, 5] / loc_centerness_mask
                    loss_center = mobula.op.BCELoss(loc_preds[:, 4], targets[:, 5]) * targets[:, 0] / num_pos_denominator
                    loss_cls = mobula.op.FocalLoss(alpha=.25, gamma=2, logits=cls_preds, targets=targets[:, 6:]) / num_pos_denominator
                    loss_total = loss_center.sum() + iou_loss.sum() + loss_cls.sum()
                    if config.TRAIN.USE_FP16:
                        with amp.scale_loss(loss_total, trainer) as scaled_losses:
                            ag.backward(scaled_losses)
                    else:
                        loss_total.backward()
                    losses_loc.append(iou_loss)
                    losses_center_ness.append(loss_center)
                    losses_cls.append(loss_cls)

            if config.use_hvd:
                trainer.step(hvd.local_size())
            else:
                trainer.step(len(ctx_list))
            if not config.use_hvd or hvd.local_rank() == 0:
                for l in losses_loc:
                    metric_loss_loc.update(None, l.sum())
                for l in losses_center_ness:
                    metric_loss_center.update(None, l.sum())
                for l in losses_cls:
                    metric_loss_cls.update(None, l.sum())
                if trainer.optimizer.num_update % config.TRAIN.log_interval == 0:  #
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

        if not config.use_hvd or hvd.local_rank() == 0:
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
    parser.add_argument('--gpus', help='The gpus used to train the network.', required=False, type=str, default="0,1,2,3")
    parser.add_argument('--hvd', help='whether training with horovod, this is useful if you have many GPUs.', action="store_true")
    parser.add_argument('--nvcc', help='', required=False, type=str, default="/usr/local/cuda-10.2/bin/nvcc")
    parser.add_argument('--im-per-gpu', help='Number of images per GPU, set this to 1 if you are facing OOM.',
                        required=False, type=int, default=3)
    parser.add_argument('--extra-flag', help='Extra flag when saving model.', required=False, type=str, default="")

    parser.add_argument('--demo', help='demo', action="store_true")
    args_known = parser.parse_known_args()[0]
    if args_known.demo:
        parser.add_argument('--demo-params', help='Params file you want to load for evaluating.', type=str, required=True)
        parser.add_argument('--viz', help='Whether visualize the results when evaluate on coco.', action="store_true")

    args = parser.parse_args()
    return args


def main():
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    # os.environ["MXNET_GPU_MEM_POOL_TYPE"] = "Round"
    load_mobula_ops()
    args = parse_args()
    setattr(mobula.config, "NVCC", args.nvcc)

    config = easydict.EasyDict()
    config.gpus = [int(x) for x in str(args.gpus).split(',')]
    config.use_hvd = args.hvd
    if config.use_hvd:
        import horovod.mxnet as hvd
        hvd.init()
        config.gpus = [hvd.local_rank()]

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
    config.TRAIN.batch_size = args.im_per_gpu * len(config.gpus)
    config.TRAIN.lr = 0.01 * config.TRAIN.batch_size / 16
    config.TRAIN.warmup_lr = config.TRAIN.lr * 0.1
    config.TRAIN.warmup_step = 1000
    config.TRAIN.wd = 1e-4
    config.TRAIN.momentum = .9
    config.TRAIN.log_interval = 100
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
    config.TRAIN.lr_step = [8, 12]
    config.TRAIN.FLIP = True
    config.TRAIN.resume = None
    config.TRAIN.trainer_resume = None
    config.TRAIN.USE_FP16 = False
    if config.TRAIN.USE_FP16:
        os.environ["MXNET_SAFE_ACCUMULATION"] = "1"
    config.network = easydict.EasyDict()
    config.network.FIXED_PARAMS = []
    config.network.use_global_stats = False
    config.network.sync_bn = True
    config.network.fpn_neck_feature_dim = 256
    if config.TRAIN.USE_FP16:
        assert config.network.sync_bn is False, "Sync BatchNorm is not supported by amp."

    config.TRAIN.log_path = "output/{}/{}-{}-{}-{}/reg_weighted_by_centerness_focal_alpha_gamma_lr_{}_{}_{}".format(
        "FCOS-res5-p5-{}".format(args.extra_flag),
        "fp16" if config.TRAIN.USE_FP16 else "fp32",
        "sync_bn" if config.network.sync_bn else "normal_bn",
        "hvd" if config.use_hvd else "",
        config.dataset.dataset_type, config.TRAIN.lr, config.TRAIN.image_short_size, config.TRAIN.image_max_long_size)

    config.val = easydict.EasyDict()
    if args.demo:
        config.val.params_file = args.demo_params
        config.val.viz = args.viz
        demo_net(config)
    else:
        os.makedirs(config.TRAIN.log_path, exist_ok=True)
        log_init(filename=os.path.join(config.TRAIN.log_path, "train_{}.log".format(time.time())))
        msg = pprint.pformat(config)
        logging.info(msg)
        train_net(config)


def inference_one_image(config, net, ctx, image_path):
    image = cv2.imread(image_path)[:, :, ::-1]
    fscale = min(config.TRAIN.image_short_size / min(image.shape[:2]), config.TRAIN.image_max_long_size / max(image.shape[:2]))
    image_padded = cv2.resize(image, (0, 0), fx=fscale, fy=fscale)
    data = mx.nd.array(image_padded[np.newaxis], ctx=ctx)
    loc_preds, cls_preds = net(data)
    bboxes_pred_list = []
    idx_start = 0
    for stride in config.FCOS.network.FPN_SCALES:
        current_layer_h = int(np.ceil(data.shape[1] / stride))
        current_layer_w = int(np.ceil(data.shape[2] / stride))
        current_layer_size = current_layer_h * current_layer_w
        loc_pred = loc_preds[:, :, idx_start:(idx_start+current_layer_size)]
        cls_pred = cls_preds[:, :, idx_start:(idx_start+current_layer_size)]
        idx_start += current_layer_size
        loc_pred = loc_pred.reshape((0, 0, current_layer_h, current_layer_w))
        cls_pred = cls_pred.reshape((0, 0, current_layer_h, current_layer_w))
        loc_pred[:, 4] = loc_pred[:, 4].sigmoid()
        cls_pred = cls_pred.sigmoid()
        rois = mobula.op.FCOSRegression(loc_pred, cls_pred, stride=stride)[0]
        rois = rois.reshape((-1, rois.shape[-1]))
        topk_indices = mx.nd.topk(rois[:, 4], k=200, ret_typ="indices")
        rois = rois[topk_indices]
        bboxes_pred_list.append(rois.asnumpy())
    bboxes_pred = np.concatenate(bboxes_pred_list, axis=0)
    if len(bboxes_pred > 0):
        cls_dets = mx.nd.contrib.box_nms(mx.nd.array(bboxes_pred, ctx=mx.cpu()),
                                         overlap_thresh=.6, coord_start=0, score_index=4, id_index=-1,
                                         force_suppress=False, in_format='corner',
                                         out_format='corner').asnumpy()
        cls_dets = cls_dets[np.where(cls_dets[:, 4] > 0.01)]
        cls_dets[:, :4] /= fscale
        if config.val.viz:
            gluoncv.utils.viz.plot_bbox(image, bboxes=cls_dets[:, :4], scores=cls_dets[:, 4], labels=cls_dets[:, 5],
                                        thresh=0.2, class_names=gluoncv.data.COCODetection.CLASSES)
            plt.show()
        cls_dets[:, 5] += 1
        return cls_dets
    else:
        return []


def demo_net(config):
    import json
    from utils.evaluate import evaluate_coco
    import tqdm
    ctx_list = [mx.gpu(x) for x in config.gpus]
    neck = PyramidNeckFCOS(feature_dim=config.network.fpn_neck_feature_dim)
    backbone = ResNetV1B(neck=neck,
                         sync_bn=config.network.sync_bn, num_devices=len(config.gpus),
                         use_global_stats=config.network.use_global_stats)
    net = FCOSFPNNet(backbone, config.dataset.NUM_CLASSES)
    net.hybridize(static_alloc=True)
    net.collect_params().load(config.val.params_file)
    net.collect_params().reset_ctx(ctx_list[0])
    results = {}
    results["results"] = []
    for x, y, names in os.walk(os.path.join(config.dataset.dataset_path, "val2017")):
        for name in tqdm.tqdm(names):
            one_img = {}
            one_img["filename"] = os.path.basename(name)
            one_img["rects"] = []
            preds = inference_one_image(config, net, ctx_list[0], os.path.join(x, name))
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
    evaluate_coco(json_label=os.path.join(config.dataset.dataset_path, "annotations/instances_val2017.json"),
                  json_predict=save_path)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
