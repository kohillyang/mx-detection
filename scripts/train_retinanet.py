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

from utils.common import log_init
from utils.blocks import FrozenBatchNorm2d
from data.bbox.bbox_dataset import AspectGroupingDataset
import mobula
from models.backbones.builder import build_backbone


def load_mobula_ops():
    mobula.op.load('RetinaNetTargetGenerator', os.path.join(os.path.dirname(__file__), "../utils/operator_cxx"))
    mobula.op.load('RetinaNetRegression', os.path.join(os.path.dirname(__file__), "../utils/operator_cxx"))
    mobula.op.load('BCELoss', os.path.join(os.path.dirname(__file__), "../utils/operator_cxx"))
    mobula.op.load("FocalLoss")


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
            # The offical RetinaNet has no GroupNorm here.
            for i in range(4):
                self.feat_cls.add(mx.gluon.nn.Conv2D(channels=256, kernel_size=3, padding=1, weight_initializer=init))
                self.feat_cls.add(mx.gluon.nn.Activation(activation="relu"))
            num_cls_channel = (num_classes - 1) * num_anchors
            self.feat_cls.add(mx.gluon.nn.Conv2D(channels=num_cls_channel, kernel_size=1, padding=0,
                                                 bias_initializer=init_bias, weight_initializer=init))

            self.feat_reg = mx.gluon.nn.HybridSequential()
            # The offical RetinaNet has no GroupNorm here.
            for i in range(4):
                self.feat_reg.add(mx.gluon.nn.Conv2D(channels=256, kernel_size=3, padding=1, weight_initializer=init))
                self.feat_reg.add(mx.gluon.nn.Activation(activation="relu"))

            self.feat_reg_loc = mx.gluon.nn.Conv2D(channels=4 * num_anchors, kernel_size=1, padding=0,
                                                   weight_initializer=init)

    def hybrid_forward(self, F, x):
        feat_reg = self.feat_reg(x)
        x_loc = self.feat_reg_loc(feat_reg)
        x_cls = self.feat_cls(x)
        return [x_loc, x_cls]


class RetinaNetFPNNet(mx.gluon.nn.HybridBlock):
    def __init__(self, backbone, num_classes, num_anchors):
        super(RetinaNetFPNNet, self).__init__()
        self.backbone = backbone
        self._head = RetinaNet_Head(num_classes, num_anchors)

    def hybrid_forward(self, F, x):
        # typically the strides are (4, 8, 16, 32, 64)
        x = self.backbone(x)
        if isinstance(x, list) or isinstance(x, tuple):
            return [self._head(xx) for xx in x]
        else:
            return [self._head(x)]


class PyramidNeckRetinaNet(mx.gluon.nn.HybridBlock):
    def __init__(self, feature_dim=256):
        super(PyramidNeckRetinaNet, self).__init__()
        self.fpn_p7_3x3 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=3, prefix="fpn_p7_3x3_", strides=2, padding=1)
        self.fpn_p6_3x3 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=3, prefix="fpn_p6_3x3_", strides=2, padding=1)
        self.fpn_p5_3x3 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=3, prefix="fpn_p5_3x3_", strides=1, padding=1)
        self.fpn_p4_3x3 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=3, prefix="fpn_p4_3x3_", strides=1, padding=1)
        self.fpn_p3_3x3 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=3, prefix="fpn_p3_3x3_", strides=1, padding=1)

        self.fpn_p5_1x1 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=1, prefix="fpn_p5_1x1_")
        self.fpn_p4_1x1 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=1, prefix="fpn_p4_1x1_")
        self.fpn_p3_1x1 = mx.gluon.nn.Conv2D(channels=feature_dim, kernel_size=1, prefix="fpn_p3_1x1_")

    def hybrid_forward(self, F, res2, res3, res4, res5):
        fpn_p5_1x1 = self.fpn_p5_1x1(res5)
        fpn_p4_1x1 = self.fpn_p4_1x1(res4)
        fpn_p3_1x1 = self.fpn_p3_1x1(res3)
        fpn_p5_upsample = F.contrib.BilinearResize2D(fpn_p5_1x1, scale_width=2, scale_height=2)
        fpn_p4_plus = F.ElementWiseSum(*[fpn_p5_upsample, fpn_p4_1x1])
        fpn_p4_upsample = F.contrib.BilinearResize2D(fpn_p4_plus, scale_width=2, scale_height=2)
        fpn_p3_plus = F.ElementWiseSum(*[fpn_p4_upsample, fpn_p3_1x1])

        P3 = self.fpn_p3_3x3(fpn_p3_plus)
        P4 = self.fpn_p4_3x3(fpn_p4_plus)
        P5 = self.fpn_p5_3x3(fpn_p5_1x1)
        # The offical RetinaNet implementation uses C5 instead of P5.
        P6 = self.fpn_p6_3x3(res5)
        P7 = self.fpn_p7_3x3(F.relu(P6))

        return P3, P4, P5, P6, P7


class RetinaNetTargetGenerator(object):
    def __init__(self, config):
        super(RetinaNetTargetGenerator, self).__init__()
        self.config = config

    def __call__(self, image_transposed, bboxes):
        # we found that generate retinanet target using cpu in dataloader is not possible.
        # For example, if the image size is (512, 512) and the anchor size is 9.
        # we need to generate a Tensor with size (84 * 9 * 64 * 64), whose size is over 3MB.
        # Instead, we move the target generating process to GPU and just pad the bbox here.
        bboxes = bboxes.copy()
        assert len(bboxes) <= self.config.dataset.max_bbox_number
        bboxes_padded = np.ones(shape=(self.config.dataset.max_bbox_number, 5)) * -1
        bboxes_padded[:len(bboxes), :5] = bboxes
        outputs = [image_transposed, bboxes]
        return outputs


def batch_fn(x):
    return x


def train_net(config):
    mx.random.seed(3)
    np.random.seed(3)

    ctx_list = [mx.gpu(x) for x in config.gpus]
    num_anchors = len(config.retinanet.network.SCALES) * len(config.retinanet.network.RATIOS)
    neck = PyramidNeckRetinaNet(feature_dim=config.network.fpn_neck_feature_dim)
    backbone = build_backbone(config, neck=neck, norm_layer=FrozenBatchNorm2d, **config.network.BACKBONE.kwargs)
    net = RetinaNetFPNNet(backbone, config.dataset.NUM_CLASSES, num_anchors)

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

    if config.TRAIN.resume is not None:
        net.collect_params().load(config.TRAIN.resume)
        logging.info("loaded resume from {}".format(config.TRAIN.resume))

    # Initialize parameters
    params = net.collect_params()
    for key in params.keys():
        if params[key]._data is None:
            if "bias" in key or "beta" in key or "offset" in key:
                default_init = mx.init.Zero()
            elif "gamma" in key or "var" in key:
                default_init = mx.init.One()
            else:
                default_init = mx.init.Xavier()

            default_init.set_verbosity(True)
            if params[key].init is not None and hasattr(params[key].init, "set_verbosity"):
                params[key].init.set_verbosity(True)
                params[key].initialize(init=params[key].init, default_init=params[key].init)
            else:
                params[key].initialize(default_init=default_init)

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
        if params_all[p].grad_req == "null" and "running" not in p:
            ignore = True
            logging.info("ignore {} because its grad req is set to null.".format(p))
        if params_fixed_prefix is not None:
            import re
            for f in params_fixed_prefix:
                if re.match(f, str(p)) is not None:
                    ignore = True
                    params_all[p].grad_req = 'null'
                    logging.info("{} is ignored when training because it matches {}.".format(p, f))
        if not ignore and params_all[p].grad_req != "null":
            params_to_train[p] = params_all[p]
    lr_steps = [len(train_loader) * int(x) for x in config.TRAIN.lr_step]
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
        for ctx in ctx_list:
            _ = net(mx.nd.random.randn(1, config.TRAIN.image_max_long_size, config.TRAIN.image_short_size, 3, ctx=ctx))
            del _
        mx.nd.waitall()
        for nbatch, data_batch in enumerate(tqdm.tqdm(train_loader, total=len(train_loader), unit_scale=1)):
            data_list = mx.gluon.utils.split_and_load(data_batch[0][0], ctx_list=ctx_list, batch_axis=0)
            targets_list = mx.gluon.utils.split_and_load(data_batch[0][1], ctx_list=ctx_list, batch_axis=0)
            losses = []
            losses_loc = []
            losses_cls = []
            with ag.record():
                for data, targets in zip(data_list, targets_list):
                    # targets: (2, 86, num_anchors, h x w)
                    fpn_predictions = net(data)
                    reg_fpn_predictions = [x[0].reshape(x[0].shape[0], -1, num_anchors, x[0].shape[2] * x[0].shape[3]) for x in fpn_predictions]
                    cls_fpn_predictions = [x[1].reshape(x[1].shape[0], -1, num_anchors, x[1].shape[2] * x[1].shape[3]) for x in fpn_predictions]
                    cls_fpn_predictions = mx.nd.concat(*cls_fpn_predictions, dim=3)
                    reg_fpn_predictions = mx.nd.concat(*reg_fpn_predictions, dim=3)
                    mask_for_cls = targets[:, 0:1]
                    mask_for_reg = targets[:, 1:2]
                    num_pos = mask_for_reg.sum()
                    num_pos = mx.nd.maximum(num_pos, mx.nd.ones_like(num_pos))
                    loss_loc = mx.nd.smooth_l1(reg_fpn_predictions - targets[:, 2:6]) * mask_for_reg / num_pos / 4
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
    parser.add_argument('--gpus', help='The gpus used to train the network.', required=False, type=str, default="0,1")
    parser.add_argument('--demo', help='demo', action="store_true")
    parser.add_argument('--nvcc', help='', required=False, type=str, default="/usr/local/cuda-10.2/bin/nvcc")

    args_known = parser.parse_known_args()[0]
    if args_known.demo:
        parser.add_argument('--demo-params', help='Params file you want to load for evaluating.', type=str, required=True)
        parser.add_argument('--viz', help='Whether visualize the results when evaluate on coco.', action="store_true")

    args = parser.parse_args()
    return args


def main():
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    os.environ["MXNET_GPU_MEM_POOL_TYPE"] = "Round"
    args = parse_args()
    setattr(mobula.config, "NVCC", args.nvcc)

    config = easydict.EasyDict()
    config.gpus = [int(x) for x in str(args.gpus).split(',')]
    config.dataset = easydict.EasyDict()
    config.dataset.NUM_CLASSES = args.num_classes
    config.dataset.dataset_type = args.dataset_type
    config.dataset.dataset_path = args.dataset_root
    config.dataset.max_bbox_number = 200

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
    config.TRAIN.lr_step = [6, 8]
    config.TRAIN.FLIP = True
    config.TRAIN.resume = None
    config.TRAIN.trainer_resume = None

    config.network = easydict.EasyDict()
    config.network.BACKBONE = easydict.EasyDict()
    config.network.BACKBONE.name = "resnetv1"
    config.network.BACKBONE.kwargs = easydict.EasyDict()
    config.network.BACKBONE.kwargs.num_layers = 50
    config.network.BACKBONE.kwargs.pretrained = True
    config.network.FIXED_PARAMS = [".*stage1.*",
                                   ".*resnetv10_conv0.*"]
    config.network.sync_bn = True
    config.network.use_global_stats = False if config.network.sync_bn else True
    config.network.fpn_neck_feature_dim = 256

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
    path = image_path
    ctx_list = [ctx]
    image = cv2.imread(path)[:, :, ::-1]
    fscale = min(config.TRAIN.image_short_size / min(image.shape[:2]), config.TRAIN.image_max_long_size / max(image.shape[:2]))
    image_padded = cv2.resize(image, (0, 0), fx=fscale, fy=fscale)
    data = mx.nd.array(image_padded[np.newaxis], ctx=ctx_list[0])
    fpn_predictions = net(data)
    bboxes_pred_list = []
    num_anchors = len(config.retinanet.network.SCALES) * len(config.retinanet.network.RATIOS)
    for reg_prediction, cls_prediction, base_size, stride in zip([x[0] for x in fpn_predictions],
                                                                 [x[1] for x in fpn_predictions],
                                                                 config.retinanet.network.BASE_SIZES,
                                                                 config.retinanet.network.FPN_STRIDES):
        reg_prediction = reg_prediction.reshape((reg_prediction.shape[0], 4, num_anchors, reg_prediction.shape[2], reg_prediction.shape[3]))
        cls_prediction = cls_prediction.reshape((cls_prediction.shape[0], cls_prediction.shape[1] // num_anchors, num_anchors, cls_prediction.shape[2], cls_prediction.shape[3]))
        cls_prediction_np = cls_prediction.sigmoid().asnumpy()
        reg_prediction_np = reg_prediction.asnumpy()
        reg_prediction_np *= np.array(config.retinanet.network.bbox_norm_coef)[None, :, None, None, None]
        print(1+1)
        rois = mobula.op.RetinaNetRegression[np.ndarray](number_of_classes=config.dataset.NUM_CLASSES,
                                                         base_size=base_size,
                                                         cls_threshold=0,
                                                         stride=stride
                                                         )(data.asnumpy(), reg_prediction_np, cls_prediction_np)
        rois = rois[0]
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
    backbone = FPNResNetV1(sync_bn=config.network.sync_bn, num_devices=len(config.gpus), use_global_stats=config.network.use_global_stats)
    ctx_list = [mx.gpu(x) for x in config.gpus]
    num_anchors = len(config.retinanet.network.SCALES) * len(config.retinanet.network.RATIOS)
    net = FCOSFPNNet(backbone, config.dataset.NUM_CLASSES, num_anchors)
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
