import copy
import pickle
import cv2
import numpy as np


def _clip_boxes(boxes, im_shape):
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def append_flipped_images(roidb):
    print ('append flipped images to roidb')
    num_images = len(roidb)
    roidb_r = copy.deepcopy(roidb)
    for i in range(num_images):
        roi_rec = roidb[i]
        boxes = roi_rec['boxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = roi_rec['width'] - oldx2 - 1
        boxes[:, 2] = roi_rec['width'] - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        entry = {'image': roi_rec['image'],
                 'height': roi_rec['height'],
                 'width': roi_rec['width'],
                 'boxes': boxes,
                 'gt_classes': roidb[i]['gt_classes'],
                 'gt_overlaps': roidb[i]['gt_overlaps'],
                 'max_classes': roidb[i]['max_classes'],
                 'max_overlaps': roidb[i]['max_overlaps'],
                 'flipped': True}

        roidb_r.append(entry)
    return roidb_r


class ROIDBDataSet(object):
    def __init__(self, db_path, transform=None, h_flip=True):
        roidb = pickle.load(open(db_path, "rb"))
        roidb = roidb if not h_flip else append_flipped_images(roidb)
        self.roidb = roidb
        self._transform = transform

    def __getitem__(self, idx):
        iroidb = copy.deepcopy(self.roidb[idx])
        image_path = iroidb["image"]
        image = cv2.imread(image_path)[:, :, ::-1]
        bbox = iroidb["boxes"]
        bbox = _clip_boxes(bbox, [image.shape[0], image.shape[1]])
        gt_classes = iroidb["gt_classes"] - 1
        bbox = np.concatenate([bbox, gt_classes[:, np.newaxis]], axis=1)
        if iroidb["flipped"]:
            image = image[:, ::-1]
        if self._transform is not None:
            return self._transform(image, bbox)
        else:
            return image, bbox

    def __len__(self):
        return len(self.roidb)
