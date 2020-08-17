from __future__ import division
import mxnet as mx
import numpy as np
import cv2


class Compose(object):
    def __init__(self, transforms=()):
        self.transforms = transforms

    def __call__(self, args):
        for trans in self.transforms:
            args = trans(args)
        return args


class ImagePad(object):
    def __init__(self, dst_shape=(368, 368)):
        self.dst_shape = dst_shape

    def __call__(self, data_dict):
        img_ori = data_dict["image"]
        bboxes = data_dict["bboxes"]
        keypoints = data_dict["keypoints"]
        availability = data_dict["availability"]
        mask_miss = data_dict["mask_miss"]

        dshape = self.dst_shape
        fscale = min(dshape[0] / img_ori.shape[0], dshape[1] / img_ori.shape[1])
        img_resized = cv2.resize(img_ori, dsize=(0, 0), fx=fscale, fy=fscale)
        img_padded = np.zeros(shape=(int(dshape[0]), int(dshape[1]), 3), dtype=np.float32)
        img_padded[:img_resized.shape[0], :img_resized.shape[1], :img_resized.shape[2]] = img_resized

        mask_miss_resized = cv2.resize(mask_miss, dsize=(0, 0), fx=fscale, fy=fscale)
        mask_miss_padded = np.zeros(shape=(int(dshape[0]), int(dshape[1])), dtype=np.float32)
        mask_miss_padded[:mask_miss_resized.shape[0], :mask_miss_resized.shape[1]] = mask_miss_resized

        keypoints = keypoints * fscale
        bboxes = bboxes * fscale

        data_dict["image"] = img_padded
        data_dict["bboxes"] = bboxes
        data_dict["keypoints"] = keypoints
        data_dict["availability"] = availability
        data_dict["mask_miss"] = mask_miss
        return data_dict


class RandomScale(object):
    def __init__(self, cfg):
        self.scale_min = cfg.TRAIN.TRANSFORM_PARAMS.scale_min
        self.scale_max = cfg.TRAIN.TRANSFORM_PARAMS.scale_max
        self.crop_size_y = cfg.TRAIN.TRANSFORM_PARAMS.crop_size_y
        self.target_dist = cfg.TRAIN.TRANSFORM_PARAMS.target_dist

    def __call__(self, data_dict):
        img_ori = data_dict["image"]
        bboxes = data_dict["bboxes"]
        keypoints = data_dict["keypoints"]
        availability = data_dict["availability"]
        mask_miss = data_dict["mask_miss"]
        bbox_idx = data_dict["crop_bbox_idx"]

        bboxes = bboxes.astype(np.float32).copy()
        keypoints = keypoints.astype(np.float32).copy()
        availability = availability.copy()

        scale_multiplier = np.random.random() * (self.scale_max - self.scale_min) + self.scale_min
        scale_self = (bboxes[bbox_idx][3] - bboxes[bbox_idx][1]) / self.crop_size_y
        scale_abs = self.target_dist / scale_self
        scale = scale_abs * scale_multiplier

        img_resized = cv2.resize(img_ori, (0, 0), fx=scale, fy=scale)
        mask_miss_resized = cv2.resize(mask_miss, (0, 0), fx=scale, fy=scale)
        bboxes[:, :4] *= scale
        keypoints *= scale

        data_dict["image"] = img_resized
        data_dict["bboxes"] = bboxes
        data_dict["keypoints"] = keypoints
        data_dict["availability"] = availability
        data_dict["mask_miss"] = mask_miss_resized
        return data_dict


class RandomSelectBBOX(object):
    def __init__(self, cfg=None):
        pass

    def __call__(self, data_dict):
        bboxes = data_dict["bboxes"]
        bbox_idx = np.random.randint(0, bboxes.shape[0])
        data_dict["crop_bbox_idx"] = bbox_idx
        return data_dict


class RandomCenterCrop(object):
    def __init__(self, cfg):
        self.center_perterb_max = cfg.TRAIN.TRANSFORM_PARAMS.center_perterb_max  # type: float
        self.crop_size_x = cfg.TRAIN.TRANSFORM_PARAMS.crop_size_x
        self.crop_size_y = cfg.TRAIN.TRANSFORM_PARAMS.crop_size_y

    def __call__(self, data_dict):
        img_ori = data_dict["image"]
        bboxes = data_dict["bboxes"]
        keypoints = data_dict["keypoints"]
        availability = data_dict["availability"]
        mask_miss = data_dict["mask_miss"]
        bbox_idx = data_dict["crop_bbox_idx"]

        bboxes = bboxes.copy()
        keypoints = keypoints.copy()
        availability = availability.copy()

        bbox = bboxes[bbox_idx]
        center_x = .5 * (bbox[0] + bbox[2])
        center_y = .5 * (bbox[1] + bbox[3])
        center_x += (np.random.random() * 2 - 1) * self.center_perterb_max
        center_y += (np.random.random() * 2 - 1) * self.center_perterb_max

        center_x = int(np.round(center_x))
        center_y = int(np.round(center_y))

        start_x = max(center_x - self.crop_size_x // 2, 0)
        start_y = max(center_y - self.crop_size_y // 2, 0)

        end_x = min(center_x + self.crop_size_x // 2, img_ori.shape[1])
        end_y = min(center_y + self.crop_size_y // 2, img_ori.shape[0])

        offset_x = center_x - self.crop_size_x // 2
        offset_y = center_y - self.crop_size_y // 2

        image_cropped = img_ori[start_y:end_y, start_x:end_x]
        image_cropped_padded = np.zeros(shape=(self.crop_size_y, self.crop_size_x, img_ori.shape[2]), dtype=np.float32)
        dst_start_x = start_x - offset_x
        dst_start_y = start_y - offset_y
        dst_end_x = dst_start_x + image_cropped.shape[1]
        dst_end_y = dst_start_y + image_cropped.shape[0]
        image_cropped_padded[dst_start_y:dst_end_y, dst_start_x:dst_end_x] = image_cropped

        mask_miss_cropped = mask_miss[start_y:end_y, start_x:end_x]
        mask_miss_cropped_padded = np.zeros(shape=(self.crop_size_y, self.crop_size_x), dtype=np.float32)
        mask_miss_cropped_padded[dst_start_y:dst_end_y, dst_start_x:dst_end_x] = mask_miss_cropped

        bboxes[:, (0, 2)] -= offset_x
        bboxes[:, (1, 3)] -= offset_y
        keypoints[:, :, 0] -= offset_x
        keypoints[:, :, 1] -= offset_y
        for m in range(keypoints.shape[0]):
            for n in range(keypoints.shape[1]):
                x, y = keypoints[m, n]
                if not (0 <= x < image_cropped_padded.shape[1] and 0 <= y < image_cropped_padded.shape[0]):
                    availability[m, n] = 0

        data_dict["image"] = image_cropped_padded
        data_dict["bboxes"] = bboxes
        data_dict["keypoints"] = keypoints
        data_dict["availability"] = availability
        data_dict["crop_bbox_idx"] = bbox_idx  # to generate mask.
        data_dict["mask_miss"] = mask_miss_cropped_padded
        return data_dict


class RandomCrop(object):
    def __init__(self, cfg):
        self.center_perterb_max = cfg.TRAIN.TRANSFORM_PARAMS.center_perterb_max  # type: float
        self.crop_size_x = cfg.TRAIN.TRANSFORM_PARAMS.crop_size_x
        self.crop_size_y = cfg.TRAIN.TRANSFORM_PARAMS.crop_size_y
        self.pad_value = cfg.TRAIN.TRANSFORM_PARAMS.PAD_VALUE

    def __call__(self, data_dict):
        img_ori = data_dict["image"]
        bboxes = data_dict["bboxes"]
        keypoints = data_dict["keypoints"]
        availability = data_dict["availability"]
        mask_miss = data_dict["mask_miss"]
        bbox_idx = data_dict["crop_bbox_idx"]

        bboxes = bboxes.copy()
        keypoints = keypoints.copy()
        availability = availability.copy()

        center_x = np.random.randint(0, img_ori.shape[1])
        center_y = np.random.randint(0, img_ori.shape[0])
        center_x = int(np.round(center_x))
        center_y = int(np.round(center_y))

        start_x = max(center_x - self.crop_size_x // 2, 0)
        start_y = max(center_y - self.crop_size_y // 2, 0)

        end_x = min(center_x + self.crop_size_x // 2, img_ori.shape[1])
        end_y = min(center_y + self.crop_size_y // 2, img_ori.shape[0])

        offset_x = center_x - self.crop_size_x // 2
        offset_y = center_y - self.crop_size_y // 2

        image_cropped = img_ori[start_y:end_y, start_x:end_x]
        image_cropped_padded = np.ones(shape=(self.crop_size_y, self.crop_size_x, img_ori.shape[2]), dtype=np.float32) * self.pad_value
        dst_start_x = start_x - offset_x
        dst_start_y = start_y - offset_y
        dst_end_x = dst_start_x + image_cropped.shape[1]
        dst_end_y = dst_start_y + image_cropped.shape[0]
        image_cropped_padded[dst_start_y:dst_end_y, dst_start_x:dst_end_x] = image_cropped

        mask_miss_cropped = mask_miss[start_y:end_y, start_x:end_x]
        mask_miss_cropped_padded = np.zeros(shape=(self.crop_size_y, self.crop_size_x), dtype=np.float32)
        mask_miss_cropped_padded[dst_start_y:dst_end_y, dst_start_x:dst_end_x] = mask_miss_cropped

        bboxes[:, (0, 2)] -= offset_x
        bboxes[:, (1, 3)] -= offset_y
        keypoints[:, :, 0] -= offset_x
        keypoints[:, :, 1] -= offset_y
        for m in range(keypoints.shape[0]):
            for n in range(keypoints.shape[1]):
                x, y = keypoints[m, n]
                if not (0 <= x < image_cropped_padded.shape[1] and 0 <= y < image_cropped_padded.shape[0]):
                    availability[m, n] = 0

        data_dict["image"] = image_cropped_padded
        data_dict["bboxes"] = bboxes
        data_dict["keypoints"] = keypoints
        data_dict["availability"] = availability
        data_dict["crop_bbox_idx"] = bbox_idx  # to generate mask.
        data_dict["mask_miss"] = mask_miss_cropped_padded
        return data_dict


def rotate_bound(image, mask, kp, angle):
    # grab the dimensions of the image and then determine the
    # center
    h, w = image.shape[:2]

    kp_new = np.empty(shape=(len(kp), 3), dtype=np.float32)
    kp_new[:, 0] = kp[:, 0]
    kp_new[:, 1] = kp[:, 1]
    kp_new[:, 2] = 1

    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    rotated = cv2.warpAffine(image, M, (nW, nH))
    mask_rotated = cv2.warpAffine(mask, M, (nW, nH))
    return rotated, mask_rotated, np.dot(kp_new, M.T), M


class RandomRotate(object):
    def __init__(self, config):
        self.min_angle = -1 * config.TRAIN.TRANSFORM_PARAMS.max_rotation_degree
        self.max_angle = config.TRAIN.TRANSFORM_PARAMS.max_rotation_degree

    def __call__(self, data_dict):
        img_ori = data_dict["image"]
        bbox = data_dict["bboxes"]
        keypoints = data_dict["keypoints"]
        availability = data_dict["availability"]
        mask_miss = data_dict["mask_miss"]
        assert bbox.shape.__len__() == 2
        assert bbox.shape[1] == 4
        assert keypoints.shape.__len__() == 3
        assert keypoints.shape[2] == 2
        assert availability.shape.__len__() == 2

        # rotate bbox and image
        kps = np.empty(shape=(len(bbox), 4, 2), dtype=np.float32)
        kps[:, 0, :] = bbox[:, (0, 1)]
        kps[:, 1, :] = bbox[:, (2, 1)]
        kps[:, 2, :] = bbox[:, (2, 3)]
        kps[:, 3, :] = bbox[:, (0, 3)]

        angle = np.random.uniform(self.min_angle, self.max_angle)
        image_rotated, mask_miss_rotated, kps_rotated, M = rotate_bound(img_ori, mask_miss, kps.reshape((-1, 2)), angle)
        kps_rotated = kps_rotated.reshape(kps.shape)
        bbox_rotated = np.zeros_like(bbox)
        bbox_rotated[:, 0] = kps_rotated[:, :, 0].min(axis=1)
        bbox_rotated[:, 1] = kps_rotated[:, :, 1].min(axis=1)
        bbox_rotated[:, 2] = kps_rotated[:, :, 0].max(axis=1)
        bbox_rotated[:, 3] = kps_rotated[:, :, 1].max(axis=1)

        # rotate keypoints
        keypoints_reshapped = keypoints.reshape(-1, 2)
        keypoints_homo = np.ones(shape=(keypoints_reshapped.shape[0], 3))
        keypoints_homo[:, :2] = keypoints_reshapped
        keypoints_rotated = keypoints_homo.dot(M.T)

        data_dict["image"] = image_rotated
        data_dict["bboxes"] = bbox_rotated
        data_dict["keypoints"] = keypoints_rotated.reshape(keypoints.shape)
        data_dict["availability"] = availability
        data_dict["mask_miss"] = mask_miss_rotated
        return data_dict


class RandomFlip(object):
    def __init__(self, flip_indices):
        self.flip_indices = flip_indices

    def __call__(self, data_dict):
        img_ori = data_dict["image"]
        bbox = data_dict["bboxes"]
        keypoints = data_dict["keypoints"]
        availability = data_dict["availability"]
        mask_miss = data_dict["mask_miss"]

        assert bbox.shape.__len__() == 2
        assert bbox.shape[1] == 4
        assert keypoints.shape.__len__() == 3
        assert keypoints.shape[2] == 2
        assert availability.shape.__len__() == 2

        if np.random.randint(0, 2):
            h, w, c = img_ori.shape
            img_flipped = img_ori[:, ::-1, :].copy()
            bbox_flipped = bbox.copy()
            keypoints_flipped = keypoints.copy()
            bbox_flipped[:, (0, 2)] = w - 1 - bbox[:, (2, 0)]
            keypoints_flipped[:, :, 0] = w - 1 - keypoints[:, :, 0]
            keypoints_flipped = keypoints_flipped[:, self.flip_indices]
            availability_flipped = availability[:, self.flip_indices]
            mask_miss_flipped = mask_miss[:, ::-1].copy()
            data_dict["image"] = img_flipped
            data_dict["bboxes"] = bbox_flipped
            data_dict["keypoints"] = keypoints_flipped
            data_dict["availability"] = availability_flipped
            data_dict["mask_miss"] = mask_miss_flipped
        return data_dict


class RandomBGRRGBInverse(object):
    def __init__(self, cfg):
        pass

    def __call__(self, data_dict):
        if np.random.random() > .5:
            data_dict["image"] = data_dict["image"][:, :, ::-1].copy()
        return data_dict


class RandomHeightScale(object):
    def __init__(self, cfg):
        self.scale_min = cfg.TRAIN.TRANSFORM_PARAMS.scale_min
        self.scale_max = cfg.TRAIN.TRANSFORM_PARAMS.scale_max
        self.crop_size_y = cfg.TRAIN.TRANSFORM_PARAMS.crop_size_y
        self.target_dist = cfg.TRAIN.TRANSFORM_PARAMS.target_dist

    def __call__(self, data_dict):
        img_ori = data_dict["image"]
        bboxes = data_dict["bboxes"]
        keypoints = data_dict["keypoints"]
        availability = data_dict["availability"]
        mask_miss = data_dict["mask_miss"]
        bbox_idx = data_dict["crop_bbox_idx"]

        bboxes = bboxes.astype(np.float32).copy()
        keypoints = keypoints.astype(np.float32).copy()
        availability = availability.copy()

        if np.random.randint(0, 2):
            scale_multiplier = np.random.random() * (self.scale_max - self.scale_min) + self.scale_min
        else:
            scale_multiplier = 1.0
        scale_self = (bboxes[bbox_idx][3] - bboxes[bbox_idx][1]) / self.crop_size_y
        scale_abs = self.target_dist / scale_self
        scale = scale_abs * scale_multiplier
        scale = min(scale, 2048.0 / img_ori.shape[0], 2048.0 / img_ori.shape[1])

        img_resized = cv2.resize(img_ori, (0, 0), fx=scale, fy=scale)
        mask_miss_resized = cv2.resize(mask_miss, (0, 0), fx=scale, fy=scale)
        bboxes[:, :4] *= scale
        keypoints *= scale

        data_dict["image"] = img_resized
        data_dict["bboxes"] = bboxes
        data_dict["keypoints"] = keypoints
        data_dict["availability"] = availability
        data_dict["mask_miss"] = mask_miss_resized
        return data_dict


class FixHeightScale(object):
    def __init__(self, cfg):
        self.scale_min = cfg.TRAIN.TRANSFORM_PARAMS.scale_min
        self.scale_max = cfg.TRAIN.TRANSFORM_PARAMS.scale_max
        self.crop_size_y = cfg.TRAIN.TRANSFORM_PARAMS.crop_size_y
        self.target_dist = cfg.TRAIN.TRANSFORM_PARAMS.target_dist

    def __call__(self, data_dict):
        img_ori = data_dict["image"]
        bboxes = data_dict["bboxes"]
        keypoints = data_dict["keypoints"]
        availability = data_dict["availability"]
        mask_miss = data_dict["mask_miss"]
        bbox_idx = data_dict["crop_bbox_idx"]

        bboxes = bboxes.astype(np.float32).copy()
        keypoints = keypoints.astype(np.float32).copy()
        availability = availability.copy()

        scale_multiplier = 1.0
        scale_self = (bboxes[bbox_idx][3] - bboxes[bbox_idx][1]) / self.crop_size_y
        scale_abs = self.target_dist / scale_self
        scale = scale_abs * scale_multiplier
        scale = min(scale, 2048.0 / img_ori.shape[0], 2048.0 / img_ori.shape[1])

        img_resized = cv2.resize(img_ori, (0, 0), fx=scale, fy=scale)
        mask_miss_resized = cv2.resize(mask_miss, (0, 0), fx=scale, fy=scale)
        bboxes[:, :4] *= scale
        keypoints *= scale

        data_dict["image"] = img_resized
        data_dict["bboxes"] = bboxes
        data_dict["keypoints"] = keypoints
        data_dict["availability"] = availability
        data_dict["mask_miss"] = mask_miss_resized
        return data_dict
