import cv2
import os
import numpy as np
import mxnet as mx
import mobula
import matplotlib.pyplot as plt


class PafHeatMapDataSet(object):
    def __init__(self, base_dataset, config, transforms=None):
        super(PafHeatMapDataSet, self).__init__()
        self.baseDataSet = base_dataset
        self.transforms = transforms
        self.number_of_keypoints = self.baseDataSet.number_of_keypoints
        self.number_of_pafs = len(self.baseDataSet.skeleton)
        mobula.op.load('HeatGen', os.path.join(os.path.dirname(__file__), "../utils/operator_cxx"))
        mobula.op.load('PAFGen', os.path.join(os.path.dirname(__file__), "../utils/operator_cxx"))
        self.cfg = config
        self.sigma = config.TRAIN.TRANSFORM_PARAMS.sigma
        self.stride = config.TRAIN.TRANSFORM_PARAMS.stride
        self.distance_threshold = config.TRAIN.TRANSFORM_PARAMS.distance_threshold

        self.idx2imageid_bboxid = []
        for i in range(len(self.baseDataSet)):
            path, bboxes, joints, image_id, mask_miss = self.baseDataSet[i]
            for j in range(len(bboxes)):
                self.idx2imageid_bboxid.append((i, j))

    def __len__(self):
        return len(self.idx2imageid_bboxid)

    def __getitem__(self, item):
        idx0, idx1 = self.idx2imageid_bboxid[item]

        path, bboxes, joints, image_id, mask_miss = self.baseDataSet[idx0]
        image = cv2.imread(path)[:, :, ::-1]
        image = image.astype(np.float32)
        keypoints = joints[:, :, :2]
        availability = np.logical_and(joints[:, :, 0] > 0, joints[:, :, 1] > 0)
        availability = availability.astype(np.float32)

        bbox_idx = 0  # 0 if transforms is None
        if self.transforms is not None:
            data_dict = {"image": image, "bboxes": bboxes, "keypoints": keypoints, "availability": availability,
                         "mask_miss": mask_miss, "crop_bbox_idx": idx1}
            data_dict = self.transforms(data_dict)
            bbox_idx = data_dict["crop_bbox_idx"] if "crop_bbox_idx" in data_dict else bbox_idx
            image = data_dict["image"]
            bboxes = data_dict["bboxes"]
            keypoints = data_dict["keypoints"]
            availability = data_dict["availability"]
            mask_miss = data_dict["mask_miss"]
        joints = np.concatenate([keypoints, availability[:, :, np.newaxis]], axis=2)
        bboxes = bboxes.astype(np.float32)
        joints = joints.astype(np.float32)
        limb_sequence = self.baseDataSet.skeleton

        heatmap = mobula.op.HeatGen[np.ndarray](self.stride, self.sigma)(image, bboxes, joints)
        pafmap = mobula.op.PAFGen[np.ndarray](limb_sequence, self.stride, self.distance_threshold)(image, bboxes, joints)
        heatmap_mask = self.genHeatmapMask(joints, heatmap, bbox_idx)
        pafmap_mask = self.genPafmapMask(limb_sequence, joints, pafmap, bbox_idx)
        mask_miss = cv2.resize(mask_miss, (heatmap.shape[2], heatmap.shape[1]))
        return image, heatmap, heatmap_mask, pafmap, pafmap_mask, mask_miss

    def genHeatmapMask(self, joints, heatmaps, bbox_idx):
        mask = np.ones_like(heatmaps)
        for j in range(len(joints[0])):
            if joints[bbox_idx, j, 2] > 0:
                pass
            else:
                mask[j][:] = 0
        return mask

    def genPafmapMask(self, limb_sequence, joints, pafmaps, bbox_idx):
        mask = np.ones_like(pafmaps)
        for j in range(len(limb_sequence)):
            if joints[bbox_idx, limb_sequence[j, 0], 2] > 0 and joints[bbox_idx, limb_sequence[j, 1], 2] > 0:
                pass
            else:
                mask[j][:] = 0
        return mask