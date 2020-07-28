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
    if config.network.USE_RFP:
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
    val_metric_5 = VOC07MApMetric(iou_thresh=.5)

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
                        class_target = fpn_label[:, 6:]

                        loc_prediction = fpn_prediction[:, :4].exp()
                        centerness_prediction = fpn_prediction[:, 4:5]
                        class_prediction = fpn_prediction[:, 5:]

                        
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
