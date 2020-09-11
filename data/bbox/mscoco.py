"""MS COCO object detection dataset."""
from __future__ import absolute_import
from __future__ import division

import os

import cv2
import mxnet as mx
from gluoncv.utils.bbox import *

from PIL import Image
import gluoncv
from .bbox_dataset import DetectionDataset

__all__ = ['COCODetection']


class COCODetection(DetectionDataset):
    CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
               'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
               'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
               'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
               'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
               'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
               'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
               'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
               'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'coco'),
                 splits=('instances_val2017',), transform=None, min_object_area=0,
                 skip_empty=True, use_crowd=True, h_flip=True):
        super(COCODetection, self).__init__()
        self.coco_dataset = gluoncv.data.COCODetection(root=root, splits=splits, use_crowd=use_crowd,
                                                       skip_empty=skip_empty, min_object_area=min_object_area)
        self.ratios = self.coco_dataset.get_im_aspect_ratio()
        self._h_flip = h_flip
        self._transformer = transform

    def at_ratio(self, idx):
        return self.ratios[idx % len(self.coco_dataset)]

    def __getitem__(self, idx):
        image, bbox = self.coco_dataset[idx % len(self.coco_dataset)]
        bbox = bbox.copy()
        if not isinstance(image, np.ndarray):
            image = image.asnumpy()
        if self._h_flip:
            if idx >= len(self.coco_dataset):
                image = image[:, ::-1, :]
                w = image.shape[1]
                bbox[:, (0, 2)] = w - 1 - bbox[:, (2, 0)]
        if self._transformer is not None:
            return self._transformer(image, bbox)
        return image, bbox

    def __len__(self):
        return len(self.coco_dataset) * 2 if self._h_flip else len(self.coco_dataset)
