"""MS COCO keypoints dataset."""
# bought from  https://raw.githubusercontent.com/dmlc/gluon-cv/master/gluoncv/data/mscoco/keypoints.py
from __future__ import absolute_import
from __future__ import division
import os
import copy
import numpy as np
import mxnet as mx
import pycocotools.mask as maskUtils
from collections import defaultdict
import cv2
from gluoncv.utils.bbox import bbox_xywh_to_xyxy, bbox_clip_xyxy


class _COCOKeyPoints(object):
    """COCO keypoint detection dataset.

    Parameters
    ----------
    root : str, default '~/mxnet/datasets/coco'
        Path to folder storing the dataset.
    splits : list of str, default ['person_keypoints_val2017']
        Json annotations name.
        Candidates can be: person_keypoints_val2017, person_keypoints_train2017.
    check_centers : bool, default is False
        If true, will force check centers of bbox and keypoints, respectively.
        If centers are far away from each other, remove this label.
    skip_empty : bool, default is False
        Whether skip entire image if no valid label is found. Use `False` if this dataset is
        for validation to avoid COCO metric error.

    """
    CLASSES = ['person']
    KEYPOINTS = {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle"
    }
    SKELETON = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
        [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'coco'),
                 splits=('person_keypoints_val2017',), check_centers=False, skip_empty=True):
        super(_COCOKeyPoints, self).__init__()
        self.num_class = len(self.CLASSES)
        self._root = os.path.expanduser(root)
        if isinstance(splits, mx.base.string_types):
            splits = [splits]
        self._splits = splits
        self._coco = []
        self._check_centers = check_centers
        self._skip_empty = skip_empty
        self.index_map = dict(zip(type(self).CLASSES, range(self.num_class)))
        self.json_id_to_contiguous = None
        self.contiguous_id_to_json = None
        self._items, self._labels = self._load_jsons()

    def __str__(self):
        detail = ','.join([str(s) for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    @property
    def classes(self):
        """Category names."""
        return type(self).CLASSES

    @property
    def num_joints(self):
        """Dataset defined: number of joints provided."""
        return 17

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]

    @property
    def coco(self):
        """Return pycocotools object for evaluation purposes."""
        if not self._coco:
            raise ValueError("No coco objects found, dataset not initialized.")
        if len(self._coco) > 1:
            raise NotImplementedError(
                "Currently we don't support evaluating {} JSON files".format(len(self._coco)))
        return self._coco[0]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_path = self._items[idx]
        img_id = int(os.path.splitext(os.path.basename(img_path))[0])

        label = copy.deepcopy(self._labels[idx])
        # img = mx.image.imread(img_path, 1)
        return img_path, label, img_id

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []
        # lazy import pycocotools
        from pycocotools.coco import COCO
        for split in self._splits:
            anno = os.path.join(self._root, 'annotations', split) + '.json'
            _coco = COCO(anno)
            self._coco.append(_coco)
            classes = [c['name'] for c in _coco.loadCats(_coco.getCatIds())]
            if not classes == self.classes:
                raise ValueError("Incompatible category names with COCO: ")
            assert classes == self.classes
            json_id_to_contiguous = {
                v: k for k, v in enumerate(_coco.getCatIds())}
            if self.json_id_to_contiguous is None:
                self.json_id_to_contiguous = json_id_to_contiguous
                self.contiguous_id_to_json = {
                    v: k for k, v in self.json_id_to_contiguous.items()}
            else:
                assert self.json_id_to_contiguous == json_id_to_contiguous
            # iterate through the annotations
            image_ids = sorted(_coco.getImgIds())
            for entry in _coco.loadImgs(image_ids):
                dirname, filename = entry['coco_url'].split('/')[-2:]
                abs_path = os.path.join(self._root, dirname, filename)
                if not os.path.exists(abs_path):
                    raise IOError('Image: {} not exists.'.format(abs_path))
                label = self._check_load_keypoints(_coco, entry)
                if not label:
                    continue

                # num of items are relative to person, not image
                for obj in label:
                    items.append(abs_path)
                    labels.append(obj)
        return items, labels

    def _check_load_keypoints(self, coco, entry):
        """Check and load ground-truth keypoints"""
        ann_ids = coco.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        width = entry['width']
        height = entry['height']

        has_valid_anno = False
        for obj1 in objs:
            def parse_obj(obj):
                contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
                if contiguous_cid >= self.num_class:
                    # not class of interest
                    return {"reason":1}
                if max(obj['keypoints']) == 0:
                    return {"reason":2}
                # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
                xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
                # require non-zero box area
                if obj['area'] <= 0 or xmax <= xmin or ymax <= ymin:
                    return {"reason":3}

                # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
                joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
                for i in range(self.num_joints):
                    joints_3d[i, 0, 0] = obj['keypoints'][i * 3 + 0]
                    joints_3d[i, 1, 0] = obj['keypoints'][i * 3 + 1]
                    # joints_3d[i, 2, 0] = 0
                    visible = min(1, obj['keypoints'][i * 3 + 2])
                    joints_3d[i, :2, 1] = visible
                    # joints_3d[i, 2, 1] = 0

                if np.sum(joints_3d[:, 0, 1]) < 1:
                    # no visible keypoint
                    return {"reason":4}

                if self._check_centers:
                    bbox_center, bbox_area = self._get_box_center_area((xmin, ymin, xmax, ymax))
                    kp_center, num_vis = self._get_keypoints_center_count(joints_3d)
                    ks = np.exp(-2 * np.sum(np.square(bbox_center - kp_center)) / bbox_area)
                    if (num_vis / 80.0 + 47 / 80.0) > ks:
                        return {"reason": 5}

                return {'bbox': (xmin, ymin, xmax, ymax), 'joints_3d': joints_3d, "reason":0}
            r = parse_obj(obj1)
            if r["reason"] == 0:
                has_valid_anno = True
            r["segmentation"] = obj1["segmentation"]
            r["image_width"] = width
            r["image_height"] = height
            valid_objs.append(r)
        if not valid_objs or not has_valid_anno:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append({
                    'bbox': np.array([-1, -1, 0, 0]),
                    'joints_3d': np.zeros((self.num_joints, 3, 2), dtype=np.float32),
                    'image_width': -1,
                    'image_height': -1,
                    'segmentation': None
                })
        if has_valid_anno:
            return valid_objs

    def _get_box_center_area(self, bbox):
        """Get bbox center"""
        c = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        area = (bbox[3]-bbox[1])*(bbox[2]-bbox[0])
        return c, area

    def _get_keypoints_center_count(self, keypoints):
        """Get geometric center of all keypoints"""
        keypoint_x = np.sum(keypoints[:, 0, 0] * (keypoints[:, 0, 1] > 0))
        keypoint_y = np.sum(keypoints[:, 1, 0] * (keypoints[:, 1, 1] > 0))
        num = float(np.sum(keypoints[:, 0, 1]))
        return np.array([keypoint_x / num, keypoint_y / num]), num


def convert_coco2openpose(joints):
    assert joints.shape[0] == 17
    assert joints.shape[1] == 3
    neck_available = int(joints[5, 2] and joints[6, 2])  # left shoulder and right shoulder
    neck_point = .5 * (joints[5, :2] + joints[6, :2]) if neck_available else np.zeros(shape=(2, ), dtype=np.float32)
    orderCOCO = np.array([1, 0, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]) - 1
    r = joints[orderCOCO]
    r[1] = neck_point[0], neck_point[1], neck_available
    return r


class COCOKeyPoints(object):
    def __init__(self, *args, **kwargs):
        self.base_dataset = _COCOKeyPoints(*args, **kwargs)
        self.objs = defaultdict(lambda : {"bboxes": [], "joints": [], "mask_miss_segs": [],
                                          "image_height": -1, "image_width":-1})
        for i in range(len(self.base_dataset)):
            image_path, label_dict, image_id = self.base_dataset[i]
            if label_dict["reason"] == 0:
                bbox = label_dict["bbox"]  # tuple as (xmin, ymin, xmax, ymax)
                joints_3d = label_dict["joints_3d"]
                xy = joints_3d[:, :2, 0]  # nx2
                visible = joints_3d[:, :1, 1]  # nx1
                joints_2d = np.concatenate([xy, visible], axis=1)  # nx3
                self.objs[image_id]["bboxes"].append(bbox)
                self.objs[image_id]["joints"].append(joints_2d)
                self.objs[image_id]["image_path"] = image_path
            else:
                self.objs[image_id]["mask_miss_segs"].append(label_dict['segmentation'])
            self.objs[image_id]["image_width"] = label_dict["image_width"]
            self.objs[image_id]["image_height"] = label_dict["image_height"]

        self.image_ids = list(self.objs.keys())
        # limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
        #            [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
        #            [1, 16], [16, 18], [3, 17], [6, 18]]
        limbSeq = [[1, 8],
                   [8, 9],
                   [9, 10],
                   [1, 11],
                   [11, 12],
                   [12, 13],
                   [1, 2],
                   [2, 3],
                   [3, 4],
                   [2, 16],
                   [1, 5],
                   [5, 6],
                   [6, 7],
                   [5, 17],
                   [1, 0],
                   [0, 14],
                   [0, 15],
                   [14, 16],
                   [15, 17]]
        self.skeleton = np.array(limbSeq)
        self.number_of_keypoints = 19
        self.flip_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        self.flip_indices = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]

    def __getitem__(self, item):
        obj = self.objs[self.image_ids[item]]
        bboxes = obj["bboxes"]
        joints = obj["joints"]
        image_path = obj["image_path"]
        # Return as path, bboxes, joints, image_id
        joints = np.array([convert_coco2openpose(x) for x in joints])
        mask_miss_segs = obj["mask_miss_segs"]
        mask_miss_binary = np.zeros((obj["image_height"], obj["image_width"]), dtype=np.int)
        for segm in mask_miss_segs:
            # seg for each instance, one seg may has several parts
            if type(segm) is list:
                # first merging them.
                rles = maskUtils.frPyObjects(segm, obj["image_height"], obj["image_width"])
                rle = maskUtils.merge(rles)
            elif type(segm['counts']) == list:
                # uncompressed RLE
                rle = maskUtils.frPyObjects(segm, obj["image_height"], obj["image_width"])
            else:
                # rle
                rle = segm
            m = maskUtils.decode(rle)
            mask_miss_binary = np.logical_or(mask_miss_binary, m)
        return image_path, np.array(bboxes), np.array(joints), self.image_ids[item], \
               np.logical_not(mask_miss_binary).astype(np.float32)

    def __len__(self):
        return len(self.image_ids)
