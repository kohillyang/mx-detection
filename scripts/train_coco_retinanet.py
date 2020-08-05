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
        num_positive_samples = 0
        for stride, base_size in zip(self.strides, self.base_sizes):
            target = mobula.op.RetinaNetTargetGenerator[np.ndarray](number_of_classes=self.number_of_classes,
                                                                    stride=stride, base_size=base_size)(
                image_transposed.astype(np.float32), bboxes.astype(np.float32))
            target[:, :, :, 1:5] /= np.array(self.bbox_norm_coef)[None, None, None]
            num_positive_samples += target[:, :, :, 1].sum()
            if self._debug_show_fig:
                axes[n_axes].imshow(target[:, :, :, 6:].max(axis=2).max(axis=2))
                n_axes += 1
            # target = np.transpose(target.reshape((target.shape[0], target.shape[1], -1)), (2, 0, 1))
            outputs.append(target)
        num_positive_samples = max(1, num_positive_samples)
        outputs.append(np.array([num_positive_samples]))
        if self._debug_show_fig:
            axes[n_axes].imshow(image_transposed.astype(np.uint8))
            gluoncv.utils.viz.plot_bbox(image_transposed, bboxes=bboxes[:, :4], ax=axes[n_axes])
            plt.show()
        outputs = tuple(mx.nd.array(x) for x in outputs)
        return outputs


def batch_fn(x):
    return x


def train_net(config):
    mx.random.seed(3)
    np.random.seed(3)

    backbone = FPNResNetV1(sync_bn=False, num_devices=4, use_global_stats=True)
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
        coco_train_dataset = COCODetection(root=config.dataset.dataset_path, splits=("instances_val2017",),
                                           h_flip=config.TRAIN.FLIP, transform=None)
        train_dataset = AspectGroupingDataset(coco_train_dataset, config,
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
    lr_steps = [len(train_dataset) // batch_size * int(x) for x in config.TRAIN.lr_step]
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

    for epoch in range(config.TRAIN.begin_epoch, config.TRAIN.end_epoch):
        net.hybridize(static_alloc=True, static_shape=False)
        for nbatch, data_batch in enumerate(tqdm.tqdm(train_loader, total=len(train_loader), unit_scale=1)):
            data_list = mx.gluon.utils.split_and_load(data_batch[0], ctx_list=ctx_list, batch_axis=0)
            label_0_list = mx.gluon.utils.split_and_load(data_batch[1], ctx_list=ctx_list, batch_axis=0)
            label_1_list = mx.gluon.utils.split_and_load(data_batch[2], ctx_list=ctx_list, batch_axis=0)
            label_2_list = mx.gluon.utils.split_and_load(data_batch[3], ctx_list=ctx_list, batch_axis=0)
            label_3_list = mx.gluon.utils.split_and_load(data_batch[4], ctx_list=ctx_list, batch_axis=0)
            label_4_list = mx.gluon.utils.split_and_load(data_batch[5], ctx_list=ctx_list, batch_axis=0)
            number_positive_list = mx.gluon.utils.split_and_load(data_batch[6], ctx_list=ctx_list, batch_axis=0)
            losses = []
            losses_loc = []
            losses_cls=[]
            with ag.record():
                for data, label0, label1, label2, label3, label4, number_positive in zip(data_list, label_0_list,
                                                                        label_1_list, label_2_list, label_3_list,
                                                                        label_4_list, number_positive_list
                                                                        ):
                    labels = [label0, label1, label2, label3, label4]
                    fpn_predictions = net(data)
                    for fpn_label, fpn_prediction in zip(labels[::-1], fpn_predictions[::-1]):
                        mask_for_cls = fpn_label[:, :, :, :, 0]
                        mask_for_reg = fpn_label[:, :, :, :, 1:2]
                        label_for_reg = fpn_label[:, :, :, :, 2:6]
                        label_for_cls = fpn_label[:, :, :, :, 6:]
                        reg_prediction = fpn_prediction[:, :4 * num_anchors, :, :].transpose((0, 2, 3, 1)).reshape_like(label_for_reg)
                        cls_prediction = fpn_prediction[:, 4 * num_anchors:, :, :].transpose((0, 2, 3, 1)).reshape_like(label_for_cls)

                        # Todo: beta should be 1/9 here, but it seems that beta can't be set...
                        loss_loc = mx.nd.smooth_l1(reg_prediction - label_for_reg) * mask_for_reg / number_positive[:, :, None, None, None]
                        loss_cls = BCEFocalLoss(cls_prediction, label_for_cls).sum(axis=4) * mask_for_cls / number_positive[:, :, None, None]
                        losses.append(loss_loc)
                        losses.append(loss_cls)

                        losses_loc.append(loss_loc)
                        losses_cls.append(loss_cls)

            ag.backward(losses)
            trainer.step(batch_size)
            for l in losses_loc:
                metric_loss_loc.update(None, l.sum())
            for l in losses_cls:
                metric_loss_cls.update(None, l.sum())
            if trainer.optimizer.num_update % config.TRAIN.log_interval == 0:
                msg = "Epoch={},Step={},lr={}, ".format(epoch, trainer.optimizer.num_update, trainer.learning_rate)
                msg += ','.join(['{}={:.3f}'.format(w, v) for w, v in zip(*eval_metrics.get())])
                logging.info(msg)
                eval_metrics.reset()

                plt.imshow(data[0].asnumpy().astype(np.uint8))
                plt.savefig(os.path.join(config.TRAIN.log_path, "{}_image.jpg".format(trainer.optimizer.num_update)))

                plt.imshow(cls_prediction[0].sigmoid().max(axis=2).max(axis=2).asnumpy())
                plt.savefig(os.path.join(config.TRAIN.log_path, "{}_heatmap.jpg".format(trainer.optimizer.num_update)))

                plt.imshow(label_for_cls[0].sigmoid().max(axis=2).max(axis=2).asnumpy())
                plt.savefig(os.path.join(config.TRAIN.log_path, "{}_heatmap_target.jpg".format(trainer.optimizer.num_update)))

                # plt.imshow(reg_prediction[0, 0, ].exp().asnumpy())
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


def main():
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    # os.environ["MXNET_GPU_MEM_POOL_TYPE"] = "Round"

    config = easydict.EasyDict()
    config.gpus = [0,1]

    config.dataset = easydict.EasyDict()
    config.dataset.NUM_CLASSES = 81  # with one background
    config.dataset.dataset_path = "/data1/coco"
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
    config.TRAIN.log_path = "output/retinanet_focal_alpha_gamma_lr_{}".format(config.TRAIN.lr)
    config.TRAIN.log_interval = 10
    config.TRAIN.cls_focal_loss_alpha = .25
    config.TRAIN.cls_focal_loss_gamma = 2
    config.TRAIN.image_short_size = 600
    config.TRAIN.image_max_long_size = 1000
    config.TRAIN.aspect_grouping = True
    # if aspect_grouping is set to False, all images will be pad to (PAD_H, PAD_W)
    config.TRAIN.PAD_H = 768
    config.TRAIN.PAD_W = 768
    config.TRAIN.begin_epoch = 0
    config.TRAIN.end_epoch = 28
    config.TRAIN.lr_step = [2, 6, 8]
    config.TRAIN.FLIP = True
    config.TRAIN.resume = None
    config.TRAIN.trainer_resume = None

    config.network = easydict.EasyDict()
    config.network.FIXED_PARAMS = []
    os.makedirs(config.TRAIN.log_path, exist_ok=True)
    log_init(filename=os.path.join(config.TRAIN.log_path, "train_{}.log".format(time.time())))
    msg = pprint.pformat(config)
    logging.info(msg)
    train_net(config)
    # demo_net(config)

def demo_net(config):
    import gluoncv
    ctx_list = [mx.gpu(0)]
    backbone = FPNResNetV1()
    net = FCOSFPNNet(backbone, config.dataset.NUM_CLASSES)
    net.collect_params().load("output/focal_alpha_gamma_lr_0.0001/0-25000.params")
    net.collect_params().reset_ctx(ctx_list[0])
    image = cv2.imread("figures/000000000785.jpg")[:, :, ::-1]
    image_padded, _ = bbox_t.ResizePad(768, 768)(image, None)
    predictions = net(mx.nd.array(image_padded[np.newaxis], ctx=ctx_list[0]))
    bboxes_pred_list = []
    for pred in predictions:
        stride = image_padded.shape[0] // pred.shape[2]

        pred[:, :4] = (pred[:, :4]).exp()
        pred[:, 5] = pred[:, 5].sigmoid()
        pred[:, 5:] = pred[:, 5:].sigmoid()

        pred_np = pred.asnumpy()
        rois = mobula.op.FCOSRegression[np.ndarray](stride)(prediction=pred_np)[0]
        rois = rois[np.where(rois[:, 4] > 0.1)]
        print(rois.shape)
        bboxes_pred_list.append(rois)
    bboxes_pred = np.concatenate(bboxes_pred_list, axis=0)
    cls_dets = mx.nd.contrib.box_nms(mx.nd.array(bboxes_pred, ctx=mx.cpu()),
                                  overlap_thresh=.3, coord_start=0, score_index=4, id_index=-1,
                                  force_suppress=True, in_format='corner',
                                  out_format='corner').asnumpy()
    cls_dets = cls_dets[np.where(cls_dets[:, 4] > 0.01)]
    # cls_dets = bboxes_pred

    gluoncv.utils.viz.plot_bbox(image_padded, bboxes=cls_dets[:, :4], scores=cls_dets[:, 4], labels=cls_dets[:, 5],
                                thresh=0)
    gluoncv.model_zoo.get_model()
    plt.show()
    pass


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
