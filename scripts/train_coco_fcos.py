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
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric

from models.fpn.resnetv1b import ResNetV1, RFPResNetV1
from utils.common import log_init
from utils.config import config, update_config
from utils.lrsheduler import WarmupMultiFactorScheduler
from utils.parallel import DataParallelModel

sys.path.append(os.path.join(os.path.dirname(__file__), "../MobulaOP"))
import mobula
mobula.op.load('FCOSTargetGenerator', os.path.join(os.path.dirname(__file__), "../utils/operator_cxx"))


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

def batch_fn(x):
    return x


class FCOS_Head(mx.gluon.nn.HybridBlock):
    def __init__(self, num_classes):
        super(FCOS_Head, self).__init__()
        with self.name_scope():
            self.feat = mx.gluon.nn.HybridSequential()
            for i in range(4):
                self.feat.add(mx.gluon.nn.Conv2D(channels=256, kernel_size=3, padding=1))
            # one extra channel for center-ness, four channel for location regression.
            # number of classes here includes one channel for the background.
            self.feat.add(mx.gluon.nn.Conv2D(channels=num_classes+1+4, kernel_size=3, padding=1))

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.feat(x)


class FCOSFPNNet(mx.gluon.nn.HybridBlock):
    def __init__(self, fpn_backbone, num_classes):
        super(FCOSFPNNet, self).__init__()
        with self.name_scope():
            self.fpn_backbone = fpn_backbone
            self.fcos_head = FCOS_Head(num_classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        # typically the strides are (4, 8, 16, 32, 64)
        return [self.fcos_head(xx) for xx in self.fpn_backbone(x)]


def train_net(ctx, begin_epoch, lr, lr_step):
    mx.random.seed(3)
    np.random.seed(3)

    batch_size = len(ctx)
    if config.network.USE_RFP and False:
        backbone = RFPResNetV1(num_devices=len(set(ctx)), num_layers=50, sync_bn=config.network.SYNC_BN,
                               pretrained=True)
    else:
        backbone = ResNetV1(num_devices=len(set(ctx)), num_layers=50, sync_bn=config.network.SYNC_BN, pretrained=True)
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
        bbox_t.FCOSTargetGenerator(config)
    ])
    val_transforms = bbox_t.Compose([
        bbox_t.Resize(target_size=config.SCALES[0][0], max_size=config.SCALES[0][1]),
        bbox_t.Normalize(),
    ])
    from data.bbox.mscoco import COCODetection
    # val_dataset = COCODetection(root=config.dataset.dataset_path, splits=("instances_val2017",), h_flip=False)
    train_dataset = COCODetection(root=config.dataset.dataset_path, splits=("instances_val2017",),
                                  h_flip=config.TRAIN.FLIP,
                                  transform=train_transforms)
    # val_dataset = YunChongDataSet(is_train=False, h_flip=False)

    # train_loader = DataLoader(train_dataset, batchsize=len(ctx))
    train_loader = mx.gluon.data.DataLoader(dataset=train_dataset, batch_size=len(ctx), batchify_fn=batch_fn,
                                            pin_memory=False, num_workers=0, last_batch="discard", shuffle=True)
    # for _ in tqdm.tqdm(train_loader, desc="Checking Dataset"):
    #     pass

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

    metric_loss_loc = mx.metric.Loss(name="loss_loc")
    metric_loss_cls = mx.metric.Loss(name="loss_cls")
    metric_loss_center = mx.metric.Loss(name="loss_center")
    eval_metrics = mx.metric.CompositeEvalMetric()
    for child_metric in [metric_loss_loc, metric_loss_cls, metric_loss_center]:
        eval_metrics.add(child_metric)

    for epoch in range(begin_epoch, config.TRAIN.end_epoch):

        for nbatch, data_batch in enumerate(tqdm.tqdm(train_loader, total=len(train_dataset) // batch_size,
                                                      unit_scale=batch_size)):
            data_batch_list = [[mx.nd.array(x).as_in_context(c) for x in d] for c, d in zip(ctx, data_batch)]
            net.collect_params().zero_grad()
            losses = []
            with ag.record():
                for data_and_label in data_batch_list:
                    image = data_and_label[0]
                    label = data_and_label[1:]
                    y_hat = net(image[np.newaxis])
                    for fpn_label, fpn_prediction in zip(label, y_hat):
                        mask = fpn_label[:, 0:1]
                        loc_target = fpn_label[:, 1:5]
                        centerness_target = fpn_label[:, 5:6]
                        class_target = fpn_label[:, 6:7]

                        loc_prediction = fpn_prediction[:, :4].exp()
                        centerness_prediction = fpn_prediction[:, 4:5]
                        class_prediction = fpn_prediction[:, 5:].log_softmax(axis=1)

                        # Todo: L1 loss -> IoU loss
                        loss_loc = mx.nd.smooth_l1(loc_prediction-loc_target, scalar=1.0) * mask / (mx.nd.sum(mask) + 1e-1)
                        loss_loc = loss_loc / 100
                        # Todo: CrossEntropy Loss-> Focal Loss
                        loss_cls = -1 * mx.nd.pick(class_prediction, index=class_target, axis=1)  / (mx.nd.sum(mask) + 1e-1)

                        # BCE loss
                        loss_centerness = BCELoss(centerness_prediction, centerness_target) * mask / (mx.nd.sum(mask) + 1e-1)

                        losses.append(loss_loc)
                        losses.append(loss_cls)
                        losses.append(loss_centerness)
            ag.backward(losses)
            trainer.step(batch_size)
            metric_loss_loc.update(None, loss_loc.sum())
            metric_loss_center.update(None, loss_centerness.sum())
            metric_loss_cls.update(None, loss_cls.sum())
            if trainer.optimizer.num_update % 10 == 0:
                msg = "Epoch={},Step={},lr={}, ".format(epoch, trainer.optimizer.num_update, trainer.learning_rate)
                msg += ','.join(['{}={:.3f}'.format(w, v) for w, v in zip(*eval_metrics.get())])
                logging.info(msg)
                eval_metrics.reset()
        save_path = "{}-{}.params".format(config.TRAIN.model_prefix, epoch)
        net.collect_params().save(save_path)
        logging.info("Saved checkpoint to {}".format(save_path))
        trainer_path = save_path + "-trainer.states"
        trainer.save_states(trainer_path)

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
