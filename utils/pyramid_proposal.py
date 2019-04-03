import numpy as np
import numpy.random as npr
from utils.nms.nms import gpu_nms_wrapper
from .bbox.bbox_transform import bbox_pred, clip_boxes
from data.transforms.generate_anchor import generate_anchors


def _filter_boxes(boxes, min_size):
    """ Remove all boxes with any side smaller than min_size """
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


def _clip_pad(tensor, pad_shape):
    """
    Clip boxes of the pad area.
    :param tensor: [n, c, H, W]
    :param pad_shape: [h, w]
    :return: [n, c, h, w]
    """
    H, W = tensor.shape[2:]
    h, w = pad_shape

    if h < H or w < W:
        tensor = tensor[:, :, :h, :w].copy()

    return tensor


def pyramid_proposal(rpn_cls_preds, rpn_bbox_preds, im_info, cfg):
    # expand values of config
    rpn_nms_threshold = cfg.TRAIN.RPN_NMS_THRESH
    rpn_pre_nms_top_n = cfg.TRAIN.RPN_PRE_NMS_TOP_N
    rpn_post_nms_top_n = cfg.TRAIN.RPN_POST_NMS_TOP_N
    rpn_min_size = cfg.TRAIN.RPN_MIN_SIZE
    feat_stride = cfg.network.RPN_FEAT_STRIDE
    num_anchors = cfg.network.NUM_ANCHORS
    scales = np.array(cfg.network.ANCHOR_SCALES)
    ratios = np.array(cfg.network.ANCHOR_RATIOS)
    im_info = im_info[0]

    nms = gpu_nms_wrapper(rpn_nms_threshold, 0)

    pre_nms_topN = rpn_pre_nms_top_n
    post_nms_topN = rpn_post_nms_top_n
    min_size = rpn_min_size

    proposal_list = []
    score_list = []
    for s, scores, bbox_deltas in zip(feat_stride, rpn_cls_preds, rpn_bbox_preds):
        stride = int(s)
        sub_anchors = generate_anchors(base_size=stride, scales=scales, ratios=ratios)
        # 1. Generate proposals from bbox_deltas and shifted anchors
        # use real image size instead of padded feature map sizes
        height, width = int(im_info[0] / stride), int(im_info[1] / stride)

        # Enumerate all shifts
        shift_x = np.arange(0, width) * stride
        shift_y = np.arange(0, height) * stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = num_anchors
        K = shifts.shape[0]
        anchors = sub_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #
        # bbox deltas will be (1, 4 * A, H, W) format
        # transpose to (1, H, W, 4 * A)
        # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
        # in slowest to fastest order
        bbox_deltas = _clip_pad(bbox_deltas, (height, width))
        bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

        # Same story for the scores:
        #
        # scores are (1, A, H, W) format
        # transpose to (1, H, W, A)
        # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
        scores = _clip_pad(scores, (height, width))
        scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_pred(anchors, bbox_deltas)

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info[:2])

        # 3. remove predicted boxes with either height or width < threshold
        # (NOTE: convert min_size to input image scale stored in im_info[2])
        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        proposal_list.append(proposals)
        score_list.append(scores)

    proposals = np.vstack(proposal_list)
    scores = np.vstack(score_list)

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    det = np.hstack((proposals, scores)).astype(np.float32)
    keep = nms(det)
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    # pad to ensure output size remains unchanged
    if len(keep) < post_nms_topN:
        pad = npr.choice(keep, size=post_nms_topN - len(keep))
        keep = np.hstack((keep, pad))
    proposals = proposals[keep, :]
    scores = scores[keep]

    # Output rois array
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
    return blob
