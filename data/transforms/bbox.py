import numpy as np
import cv2
import mxnet.ndarray as nd
import mxnet as mx
import mobula

def rotate_nobound(image, kp, angle, center=None, scale=1.):
    (h, w) = image.shape[:2]
    kp_new = np.empty(shape=(len(kp), 3), dtype=np.float32)
    kp_new[:, 0] = kp[:, 0]
    kp_new[:, 1] = kp[:, 1]
    kp_new[:, 2] = 1

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    kp_new[:, 0:2] = np.dot(kp_new, M.T)

    kp_new[:, 0] = kp_new[:, 0]
    kp_new[:, 1] = kp_new[:, 1]
    kp_new[:, 2] = 1

    kp_new[(kp_new[:, 0] < 0) | (kp_new[:, 1] < 0), 2] = -1

    return rotated, kp_new[:, :2]


def rotate_bound(image, kp, angle):
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

    kp_new[:, 0:2] = np.dot(kp_new, M.T)

    kp_new[:, 0] = kp_new[:, 0]
    kp_new[:, 1] = kp_new[:, 1]
    kp_new[:, 2] = 1

    kp_new[(kp_new[:, 0] < 0) | (kp_new[:, 1] < 0), 2] = -1

    mean_value = image.mean(0).mean(0)
    rotated = cv2.warpAffine(image, M, (nW, nH))

    return rotated, kp_new[:, :2]


class RandomRotate(object):
    def __init__(self, bound=True, min_angle=15, max_angle=15):
        self.bound = bound
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, image, bbox):
        kps = np.empty(shape=(len(bbox), 4, 2), dtype=np.float32)
        kps[:, 0, :] = bbox[:, (0, 1)]
        kps[:, 1, :] = bbox[:, (2, 1)]
        kps[:, 2, :] = bbox[:, (2, 3)]
        kps[:, 3, :] = bbox[:, (0, 3)]
        kps_shape = kps.shape
        kps = kps.reshape((-1, 2))
        angle = np.random.uniform(self.min_angle, self.max_angle)
        if self.bound:
            img, kps = rotate_bound(image, kps, angle)
        else:
            img, kps = rotate_nobound(image, kps, angle)
        kps = kps.reshape(kps_shape)
        bbox[:, 0] = kps[:, :, 0].min(axis=1)
        bbox[:, 1] = kps[:, :, 1].min(axis=1)
        bbox[:, 2] = kps[:, :, 0].max(axis=1)
        bbox[:, 3] = kps[:, :, 1].max(axis=1)
        return img, bbox


class RandomHFlip(object):

    def __call__(self, image, bbox):
        from random import randint
        if randint(0, 1):
            image = image[:, ::-1, :]
            w = image.shape[1]
            bbox[:, (0, 2)] = w - 1 - bbox[:, (2, 0)]
        return image, bbox


class RandomRotate(object):
    def __init__(self, bound=True, min_angle=15, max_angle=15):
        self.bound = bound
        self.min_angle = min_angle
        self.max_angle = max_angle

    def __call__(self, image, bbox):
        kps = np.empty(shape=(len(bbox), 4, 2), dtype=np.float32)
        kps[:, 0, :] = bbox[:, (0, 1)]
        kps[:, 1, :] = bbox[:, (2, 1)]
        kps[:, 2, :] = bbox[:, (2, 3)]
        kps[:, 3, :] = bbox[:, (0, 3)]
        kps_shape = kps.shape
        kps = kps.reshape((-1, 2))
        angle = np.random.uniform(self.min_angle, self.max_angle)
        if self.bound:
            img, kps = rotate_bound(image, kps, angle)
        else:
            img, kps = rotate_nobound(image, kps, angle)
        kps = kps.reshape(kps_shape)
        bbox[:, 0] = kps[:, :, 0].min(axis=1)
        bbox[:, 1] = kps[:, :, 1].min(axis=1)
        bbox[:, 2] = kps[:, :, 0].max(axis=1)
        bbox[:, 3] = kps[:, :, 1].max(axis=1)
        return img, bbox


class Resize(object):
    def __init__(self, target_size=1024, max_size=1024):
        self.target_size = target_size
        self.max_size = max_size

    def __call__(self, image, bbox=None):
        im_shape = image.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(self.target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > self.max_size:
            im_scale = float(self.max_size) / float(im_size_max)
        im = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)
        pad = lambda x: x if x % 64 == 0 else x + 64 - x % 64
        im_padded = np.zeros(shape=(pad(im.shape[0]), pad(im.shape[1]), im.shape[2]), dtype=np.float32)
        im_padded[:im.shape[0], :im.shape[1], :] = im
        if bbox is not None and len(bbox) > 1:
            bbox = bbox.astype('f')
            bbox[:, :4] *= im_scale
        return im_padded, bbox


class RandomResize(object):
    def __init__(self, scales):
        """
        :param scales: A list of tuple indicating the target_size and max_size, eg. [(600, 800), (700, 1000)]
        """
        self.scales = scales

    def __call__(self, image, bbox=None):
        size_idx = np.random.choice(list(range(len(self.scales))))
        target_size, max_size = self.scales[size_idx]
        im_shape = image.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(image, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)
        if bbox is not None and len(bbox) > 1:
            bbox = bbox.astype('f')
            bbox[:, :4] *= im_scale
        return im, bbox


class Normalize(object):
    def __call__(self, image, bbox=None):
        # Here we only transpose and expand the axes.
        image = image.transpose((2, 0, 1))[np.newaxis]
        if bbox is not None:
            bbox = bbox[np.newaxis]
        return image, bbox


class AddImInfo(object):
    def __init__(self, pad_n=0):
        self.pad_n = pad_n

    def __call__(self, image, bbox=None):
        im_info = mx.nd.array([[image.shape[2], image.shape[3], 1]], ctx=mx.cpu(), dtype='f')
        if self.pad_n > 0:
            n, c, h, w = image.shape
            h_padded = h if h % self.pad_n == 0 else h + self.pad_n - h % self.pad_n
            w_padded = w if w % self.pad_n == 0 else w + self.pad_n - w % self.pad_n
            img_padded = mx.nd.zeros(shape=(n, c, h_padded, w_padded), dtype=image.dtype)
            img_padded[:n, :c, :h, :w] = image
        return img_padded, im_info, bbox


class DeNormalize(object):
    def __init__(self, mean=(0, 0, 0), std=(1, 1, 1)):
        self.mean = mean
        self.std = std

    def __call__(self, image, bbox, kps):
        image = image.transpose((1, 2, 0))
        image = image * np.array(self.std)[np.newaxis, np.newaxis]
        image = image + np.array(self.mean)[np.newaxis, np.newaxis]
        return image, bbox, kps


class GenSegLabel(object):
    def __call__(self, image, bbox, kps):
        seg_label = np.zeros(shape=(len(kps), image.shape[0], image.shape[1]), dtype=np.float32)
        for n in range(len(kps)):
            cv2.fillPoly(seg_label[n], kps[n:(n + 1)].astype('i'), color=(1, 1, 1))
        return image, bbox, seg_label


class AssignAnchor(object):
    def __init__(self, cfg, feat_strides=(16, 16), symbol=None):
        self.cfg = cfg
        self.feat_strides = feat_strides
        self.allowed_border = 0
        self.symbol = symbol  # type: mx.sym.Symbol

    def __call__(self, image_transposed, bbox):
        from utils.rpn import assign_anchor

        gt_boxes = bbox.copy()
        assert gt_boxes.shape[0] == 1
        gt_boxes = gt_boxes[0]

        # The class index of background in Faster RCNN network is 0, however, there is no
        # background class in gluon-cv.
        gt_boxes[:, 4] += 1

        h_padded, w_padded = image_transposed.shape[2:4]
        im_info = np.array([[h_padded, w_padded, 1.0]])
        data = image_transposed.copy()

        # Assign Anchor
        if self.symbol is not None:
            arg_shapes, out_shapes, aux_shapes = self.symbol.infer_shape(data=data.shape)
            assert len(out_shapes) == 1
            feat_shape = out_shapes[0]
        else:
            feat_shape = [1, 2 * self.cfg.network.NUM_ANCHORS * 2,
                          data.shape[2] // self.feat_strides[0],
                          data.shape[3] // self.feat_strides[1]]
        label_dict = assign_anchor(feat_shape, gt_boxes[:, :5], im_info, self.cfg,
                                   self.feat_strides[0], self.cfg.network.ANCHOR_SCALES,
                                   self.cfg.network.ANCHOR_RATIOS, self.allowed_border)

        # Convert them to NDArray
        data = nd.array(data)
        im_info = nd.array(im_info)
        gt_boxes = nd.array(gt_boxes[np.newaxis, :, :5])
        label = nd.array(label_dict["label"])
        bbox_target = nd.array(label_dict["bbox_target"])
        bbox_weight = nd.array(label_dict["bbox_weight"])
        return data, im_info, gt_boxes, label, bbox_target, bbox_weight


class AssignPyramidAnchor(object):
    def __init__(self, cfg, symbol, pad_n=32, restrict = False):
        self.cfg = cfg
        self.allowed_border = np.inf
        self.symbol = symbol  # type: mx.sym.Symbol
        self.pad_n = pad_n
        self.restrict = restrict

    def __call__(self, image_transposed, bbox):
        from utils.rpn import assign_pyramid_anchor

        gt_boxes = bbox.copy().astype('f')

        # The class index of background in Faster R-CNN is 0, however, there is no
        # background class in gluon-cv.
        gt_boxes[:, :, 4] += 1

        if self.restrict:
            assert np.all(gt_boxes[:, :, 2] > gt_boxes[:, :, 0])
            assert np.all(gt_boxes[:, :, 3] > gt_boxes[:, :, 1])
            assert np.all(gt_boxes[:, :, (2, 0)] >= 0)
            assert np.all(gt_boxes[:, :, (2, 0)] < image_transposed.shape[3])
            assert np.all(gt_boxes[:, :, (3, 1)] >= 0)
            assert np.all(gt_boxes[:, :, (3, 1)] < image_transposed.shape[2])

        h_padded, w_padded = image_transposed.shape[2:4]
        im_info = np.array([[h_padded, w_padded, 1.0]])
        data = image_transposed.copy().astype('f')

        if self.pad_n > 0:
            n, c, h, w = data.shape
            h_padded = h if h % self.pad_n == 0 else h + self.pad_n - h % self.pad_n
            w_padded = w if w % self.pad_n == 0 else w + self.pad_n - w % self.pad_n
            img_padded = mx.nd.zeros(shape=(n, c, h_padded, w_padded), dtype=data.dtype)
            img_padded[:n, :c, :h, :w] = data
            data = img_padded
        # Assign Anchor
        feat_shapes = [x.infer_shape(data=data.shape)[1] for x in self.symbol]

        label_dict = assign_pyramid_anchor(feat_shapes,
                                           gt_boxes=gt_boxes[0],
                                           im_info=im_info,
                                           cfg=self.cfg,
                                           feat_strides=self.cfg.network.RPN_FEAT_STRIDE,
                                           scales=self.cfg.network.ANCHOR_SCALES,
                                           ratios=self.cfg.network.ANCHOR_RATIOS,
                                           allowed_border=np.inf)
        gt_boxes = nd.array(gt_boxes[:, :, :5])
        if gt_boxes.size == 0:
            gt_boxes = nd.zeros(shape=(1, 1), dtype='f')
        # Convert them to NDArray
        data = nd.array(data)
        im_info = nd.array(im_info)
        label = nd.array(label_dict["label"])
        bbox_target = nd.array(label_dict["bbox_target"])
        bbox_weight = nd.array(label_dict["bbox_weight"])
        return data, im_info, gt_boxes, label, bbox_target, bbox_weight


class FCOSTargetGenerator(object):
    def __init__(self, config):
        super(FCOSTargetGenerator, self).__init__()
        self.config = config
        self.strides = config.FCOS.network.FPN_SCALES
        self.fpn_min_distance = config.FCOS.network.FPN_MINIMUM_DISTANCES
        self.fpn_max_distance = config.FCOS.network.FPN_MAXIMUM_DISTANCES
        self.number_of_classes = config.dataset.NUM_CLASSES

    def __call__(self, image_transposed, bboxes):
        h, w, c = image_transposed.shape
        for s in self.strides:
            assert h % s == 0
            assert w % s == 0
        outputs = [image_transposed]
        for stride, min_distance, max_distance in zip(self.strides, self.fpn_min_distance, self.fpn_max_distance):
            target = mobula.op.FCOSTargetGenerator[np.ndarray](stride, min_distance, max_distance, self.number_of_classes)(
                image_transposed.astype(np.float32), bboxes.astype(np.float32))
            target = target.transpose((2, 0, 1))
            target = target[np.newaxis]
            outputs.append(target)
        return outputs


class Compose(object):
    def __init__(self, transforms=()):
        self.transforms = transforms

    def __call__(self, *args):
        for trans in self.transforms:
            args = trans(*args)
        return args
