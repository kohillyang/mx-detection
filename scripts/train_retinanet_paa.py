from __future__ import print_function

import logging
import os

os.environ["MXNET_ENGINE_TYPE"] = "NaiveEngine"

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

import mobula


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
        return [x_loc, x_cls]


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


class RetinaNetTargetPadding(object):
    def __init__(self, config):
        super(RetinaNetTargetPadding, self).__init__()
        self.max_gt_boxes_num = config.TRAIN.max_gt_boxes_num

    def __call__(self, image_transposed, bboxes):
        assert len(bboxes) <= self.max_gt_boxes_num
        bboxes = bboxes.copy()
        # bboxes[:, 4] += 1
        bboxes_padded = np.zeros(shape=(self.max_gt_boxes_num, bboxes.shape[1]))
        bboxes_padded[:len(bboxes)] = bboxes
        outputs = (image_transposed, bboxes_padded, np.array([len(bboxes)]))
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
    from utils import graph_optimize
    net = FCOSFPNNet(backbone, config.dataset.NUM_CLASSES, num_anchors)
    # if config.network.merge_backbone_bn:
    #     net = graph_optimize.merge_gluon_hybrid_block_bn(net, (1, 368, 368, 3))

    # Resume parameters.
    resume = "pretrained/retinanet/8-325000.params"
    if resume is not None:
        params_coco = mx.nd.load(resume)
        params_local_mapped = {}
        for k in params_coco:
            params_local_mapped[k.replace("arg:", "").replace("aux:", "").replace("sync", "")] = params_coco[k]

        params = net.collect_params()

        for k in params.keys():
            try:
                pp = params_local_mapped[k]
                params[k]._load_init(pp, ctx=mx.cpu())
                print("success load {}".format(k))
            except Exception as e:
                logging.exception(e)

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
    if config.TRAIN.resume is not None:
        net.collect_params().load(config.TRAIN.resume)
        logging.info("loaded resume from {}".format(config.TRAIN.resume))

    net.collect_params().reset_ctx(list(set(ctx_list)))

    from data.bbox.mscoco import COCODetection
    if config.TRAIN.aspect_grouping:
        if config.dataset.dataset_type == "coco":
            from data.bbox.mscoco import COCODetection
            base_train_dataset = COCODetection(root=config.dataset.dataset_path, splits=("instances_val2017",),
                                               h_flip=config.TRAIN.FLIP, transform=None)
        elif config.dataset.dataset_type == "voc":
            from data.bbox.voc import VOCDetection
            base_train_dataset = VOCDetection(root=config.dataset.dataset_path,
                                              splits=((2007, 'trainval'), (2012, 'trainval')),
                                              preload_label=False)
        else:
            assert False
        train_dataset = AspectGroupingDataset(base_train_dataset, config,
                                              target_generator=RetinaNetTargetPadding(config))
        train_loader = mx.gluon.data.DataLoader(dataset=train_dataset, batch_size=1, batchify_fn=batch_fn,
                                                num_workers=0, last_batch="discard", shuffle=True, thread_pool=False)
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
    mobula.op.load('PAAScore', os.path.join(os.path.dirname(__file__), "../utils/operator_cxx"))
    mobula.op.load('RetinaNetRegression', os.path.join(os.path.dirname(__file__), "../utils/operator_cxx"))

    for epoch in range(config.TRAIN.begin_epoch, config.TRAIN.end_epoch):
        # net.hybridize(static_alloc=True, static_shape=False)
        # for ctx in ctx_list:
        #     _ = net(mx.nd.random.randn(1, config.TRAIN.image_max_long_size, config.TRAIN.image_short_size, 3, ctx=ctx))
        #     del _
        # mx.nd.waitall()
        for nbatch, data_batch in enumerate(tqdm.tqdm(train_loader, total=len(train_loader), unit_scale=1)):
            data_list = mx.gluon.utils.split_and_load(mx.nd.array(data_batch[0][0]), ctx_list=ctx_list, batch_axis=0)
            gt_boxes_list = mx.gluon.utils.split_and_load(mx.nd.array(data_batch[0][1]), ctx_list=ctx_list, batch_axis=0)
            gt_boxes_number_list = mx.gluon.utils.split_and_load(mx.nd.array(data_batch[0][2]), ctx_list=ctx_list, batch_axis=0)
            losses = []
            losses_loc = []
            losses_cls = []
            # with ag.record():
            for data, gt_boxes, gt_boxes_number in zip(data_list, gt_boxes_list, gt_boxes_number_list):
                # targets: (2, 86, num_anchors, h x w)
                fpn_predictions = net(data)
                import gluoncv
                fig, axes = plt.subplots(5, 2, squeeze=False)
                nlayer = 0
                print(gt_boxes[0])
                topk_scores_list = []
                for stride, base_size, (reg_pred, cls_pred) in zip(config.retinanet.network.FPN_STRIDES,
                                                                   config.retinanet.network.BASE_SIZES,
                                                                   fpn_predictions):
                    # cls_pred_reshapped = cls_pred.reshape((cls_pred.shape[0], -1))
                    # cls_topk_value, cls_topk_indices = mx.nd.topk(cls_pred_reshapped, axis=1, ret_typ="both", k=1000)
                    scales = config.retinanet.network.SCALES
                    ratios = config.retinanet.network.RATIOS
                    anchors = [[(s * np.sqrt(r), s * np.sqrt(1 / r)) for s in scales] for r in ratios]
                    anchors_base_wh = np.array(anchors) * np.array(base_size)[np.newaxis, np.newaxis, :]
                    anchors_base_wh = anchors_base_wh.reshape(-1, 2)
                    anchors_base_wh = mx.nd.array(anchors_base_wh).as_in_context(data.context)

                    reg_prediction = reg_pred[0:1]
                    cls_prediction = cls_pred[0:1]
                    reg_prediction = reg_pred.reshape((reg_prediction.shape[0], 4, num_anchors, reg_prediction.shape[2], reg_prediction.shape[3]))
                    cls_prediction = cls_pred.reshape((cls_prediction.shape[0], cls_prediction.shape[1] // num_anchors, num_anchors,
                                                             cls_prediction.shape[2], cls_prediction.shape[3]))
                    cls_prediction_np = cls_prediction.sigmoid().asnumpy()
                    reg_prediction_np = reg_prediction.asnumpy()
                    reg_prediction_np *= np.array(config.retinanet.network.bbox_norm_coef)[None, :, None, None, None]
                    rois = mobula.op.RetinaNetRegression[np.ndarray](number_of_classes=config.dataset.NUM_CLASSES,
                                                                     base_size=base_size,
                                                                     cls_threshold=0,
                                                                     stride=stride
                                                                     )(data.asnumpy(), reg_prediction_np,
                                                                       cls_prediction_np)

                    rois = rois[0]
                    rois = rois[np.argsort(-1 * rois[:, 4])[:200]]
                    rois = rois[np.where(rois[:, 4] > .1)]
                    axes = axes.reshape((5, 2))

                    gluoncv.utils.viz.plot_bbox(data[0], rois[:, :4], rois[:, 4], rois[:, 5], ax=axes[nlayer, 0], class_names=gluoncv.data.COCODetection.CLASSES)

                    print(cls_pred.sigmoid().max())
                    bbox_norm_coef = mx.nd.array(config.retinanet.network.bbox_norm_coef).as_in_context(mx.cpu())
                    paa_scores_and_gt_indices = mobula.op.PAAScore(data[0:1].as_in_context(mx.cpu()),
                                                                   reg_pred[0:1].as_in_context(mx.cpu()),
                                                                   cls_pred[0:1].sigmoid().as_in_context(mx.cpu()),
                                                                   anchors_base_wh.as_in_context(mx.cpu()),
                                                                   gt_boxes[0:1].as_in_context(mx.cpu()),
                                                                   gt_boxes_number[0:1].as_in_context(mx.cpu()),
                                                                   bbox_norm_coef,
                                                                   stride=stride)
                    paa_scores = paa_scores_and_gt_indices[:, 0]
                    try:
                        topk_scores = paa_scores[0].reshape(-1)[mx.nd.topk(paa_scores[0].reshape(-1), ret_typ="indices", k=25)].asnumpy()
                        topk_scores_list.append(topk_scores)
                    except Exception:
                        pass

                    nlayer += 1
            topk_scores = np.concatenate(topk_scores_list, axis=0)
            topk_scores = topk_scores[np.where(topk_scores > 0)]
            axes[0, 1].hist(topk_scores, 25)

            plt.show()

            # ag.backward(losses)
            # trainer.step(len(ctx_list))
            # for l in losses_loc:
            #     metric_loss_loc.update(None, l.sum())
            # for l in losses_cls:
            #     metric_loss_cls.update(None, l.sum())
            # if trainer.optimizer.num_update % config.TRAIN.log_interval == 0:
            #     msg = "Epoch={},Step={},lr={}, ".format(epoch, trainer.optimizer.num_update, trainer.learning_rate)
            #     msg += ','.join(['{}={:.3f}'.format(w, v) for w, v in zip(*eval_metrics.get())])
            #     logging.info(msg)
            #     eval_metrics.reset()
            #
            # if trainer.optimizer.num_update % 5000 == 0:
            #     save_path = os.path.join(config.TRAIN.log_path, "{}-{}.params".format(epoch, trainer.optimizer.num_update))
            #     net.collect_params().save(save_path)
            #     logging.info("Saved checkpoint to {}".format(save_path))
            #     trainer_path = save_path + "-trainer.states"
            #     trainer.save_states(trainer_path)
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
    parser.add_argument('--nvcc', help='', required=False, type=str, default="/usr/local/cuda-10.1/bin/nvcc")

    args_known = parser.parse_known_args()[0]
    if args_known.demo:
        parser.add_argument('--demo-params', help='Params file you want to load for evaluating.', type=str, required=True)
        parser.add_argument('--viz', help='Whether visualize the results when evaluate on coco.', action="store_true")

    args = parser.parse_args()
    return args


def main():
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    # os.environ["MXNET_GPU_MEM_POOL_TYPE"] = "Round"
    args = parse_args()
    setattr(mobula.config, "NVCC", args.nvcc)
    mobula.config.SHOW_BUILDING_COMMAND = True
    mobula.config.USING_ASYNC_EXEC = False
    mobula.config.HOST_NUM_THREADS = 1
    mobula.config.DEBUG=True
    mobula.config.USING_OPTIMIZATION=False
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
    config.TRAIN.lr_step = [6, 8]
    config.TRAIN.FLIP = True
    config.TRAIN.resume = None
    config.TRAIN.trainer_resume = None

    # max number of gt boxes, all gt_boxes will be padded to this length.
    config.TRAIN.max_gt_boxes_num = 200
    config.TRAIN.top_k_per_fpn_layer = 200

    config.network = easydict.EasyDict()
    config.network.FIXED_PARAMS = []
    config.network.sync_bn = False
    config.network.use_global_stats = False if config.network.sync_bn else True
    config.network.merge_backbone_bn = False

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
    num_reg_channels = 4 * num_anchors
    cls_fpn_predictions = [
        x[:, num_reg_channels:].reshape(x.shape[0], -1, num_anchors, x.shape[2], x.shape[3]).sigmoid()
        for x in fpn_predictions]
    reg_fpn_predictions = [x[:, :num_reg_channels].reshape(x.shape[0], -1, num_anchors, x.shape[2], x.shape[3])
                           for x in fpn_predictions]
    for reg_prediction, cls_prediction, base_size, stride in zip(reg_fpn_predictions,
                                                                 cls_fpn_predictions,
                                                                 config.retinanet.network.BASE_SIZES,
                                                                 config.retinanet.network.FPN_STRIDES):
        fpn_prediction = mx.nd.concat(reg_prediction, cls_prediction, dim=1)
        fpn_prediction_reshaped_np = fpn_prediction.transpose((0, 3, 4, 2, 1)).asnumpy()
        fpn_prediction_reshaped_np[:, :, :, :, :4] *= np.array(config.retinanet.network.bbox_norm_coef)[None, None, None, None]
        rois = mobula.op.RetinaNetRegression[np.ndarray](number_of_classes=config.dataset.NUM_CLASSES,
                                                         base_size=base_size,
                                                         cls_threshold=0,
                                                         stride=stride
                                                         )(image=data.asnumpy(),
                                                           feature=fpn_prediction_reshaped_np)
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
