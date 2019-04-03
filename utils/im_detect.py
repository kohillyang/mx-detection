import cv2
import mxnet.ndarray as nd
import numpy as np
import mxnet as mx
from utils.bbox import bbox_pred, clip_boxes


def pre_compute_deltas(bbox_deltas, bbox_stds):
    n_rois = bbox_deltas.shape[0]
    bbox_deltas = bbox_deltas.reshape((n_rois, -1, 4)) * np.array(bbox_stds)[np.newaxis, np.newaxis]
    return bbox_deltas.reshape((n_rois, -1))


def im_detect_bbox_aug(net, im, scales, ctx, bbox_stds=(1, 1, 1, 1),
                       hflip=False, vflip=False, vhflip=False,
                       threshold=1e-3, nms_threshold=.3,
                       viz=False, pad=False, class_agnostic=True):
    all_bboxes = []
    all_scores = []
    img_ori = im.copy()
    for scale_min, scale_max in scales:
        fscale = min(1.0 * scale_min / min(img_ori.shape[:2]), 1.0 * scale_max / max(img_ori.shape[:2]))
        img_resized = cv2.resize(img_ori, (0, 0), fx=fscale, fy=fscale, interpolation=cv2.INTER_CUBIC)
        h, w, c = img_resized.shape
        if pad:
            h_padded = h if h % 32 == 0 else h + 32 - h % 32
            w_padded = w if w % 32 == 0 else w + 32 - w % 32
        else:
            h_padded = h
            w_padded = w

        img_padded = np.zeros(shape=(h_padded, w_padded, c), dtype=img_resized.dtype)
        img_padded[:h, :w, :] = img_resized

        data = nd.array(img_padded.transpose(2, 0, 1)[np.newaxis], ctx=ctx[0])
        im_info = nd.array([[h, w, 1.0]], ctx=ctx[0])

        rois, scores, bbox_deltas = net(data, im_info)
        rois = rois[:, 1:]
        if isinstance(rois, mx.nd.NDArray):
            rois = rois.asnumpy()
        bbox_deltas = bbox_deltas[0].asnumpy()
        bbox_deltas = pre_compute_deltas(bbox_deltas, bbox_stds=bbox_stds)
        bbox = bbox_pred(rois, bbox_deltas)
        bbox = clip_boxes(bbox, data.shape[2:4])
        bbox /= fscale
        all_bboxes.append(bbox)
        all_scores.append(scores[0].asnumpy())

        # flip
        if hflip:
            rois, scores, bbox_deltas = net(data[:, :, :, ::-1], im_info)
            if isinstance(rois, mx.nd.NDArray):
                rois = rois.asnumpy()
            rois = rois[:, 1:]
            bbox_deltas = bbox_deltas[0].asnumpy()
            bbox_deltas = pre_compute_deltas(bbox_deltas, bbox_stds=bbox_stds)
            bbox = bbox_pred(rois, bbox_deltas)
            bbox = clip_boxes(bbox, data.shape[2:4])

            tmp = bbox[:, 0::4].copy()
            bbox[:, 0::4] = data.shape[3] - bbox[:, 2::4]  # x0 = w - x0
            bbox[:, 2::4] = data.shape[3] - tmp  # x1 = w -x1
            bbox /= fscale

            all_bboxes.append(bbox)
            all_scores.append(scores[0].asnumpy())

        if vflip:
            rois, scores, bbox_deltas = net(data[:, :, ::-1, :], im_info)
            if isinstance(rois, mx.nd.NDArray):
                rois = rois.asnumpy()
            rois = rois[:, 1:]
            bbox_deltas = bbox_deltas[0].asnumpy()
            bbox_deltas = pre_compute_deltas(bbox_deltas, bbox_stds=bbox_stds)
            bbox = bbox_pred(rois, bbox_deltas)
            bbox = clip_boxes(bbox, data.shape[2:4])

            tmp = bbox[:, 1::4].copy()
            bbox[:, 1::4] = data.shape[2] - bbox[:, 3::4]  # y0 = h - y1
            bbox[:, 3::4] = data.shape[2] - tmp  # y1 = h -y0
            bbox /= fscale

            all_bboxes.append(bbox)
            all_scores.append(scores[0].asnumpy())

        if vhflip:
            rois, scores, bbox_deltas = net(data[:, :, ::-1, ::-1], im_info)
            if isinstance(rois, mx.nd.NDArray):
                rois = rois.asnumpy()
            rois = rois[:, 1:]
            bbox_deltas = bbox_deltas[0].asnumpy()
            bbox_deltas = pre_compute_deltas(bbox_deltas, bbox_stds=bbox_stds)
            bbox = bbox_pred(rois, bbox_deltas)
            bbox = clip_boxes(bbox, data.shape[2:4])

            tmp = bbox[:, 0::4].copy()
            bbox[:, 0::4] = data.shape[3] - bbox[:, 2::4]  # x0 = w - x0
            bbox[:, 2::4] = data.shape[3] - tmp  # x1 = w -x1
            tmp = bbox[:, 1::4].copy()
            bbox[:, 1::4] = data.shape[2] - bbox[:, 3::4]  # y0 = h - y1
            bbox[:, 3::4] = data.shape[2] - tmp  # y1 = h -y0

            bbox /= fscale

            all_bboxes.append(bbox)
            all_scores.append(scores[0].asnumpy())

    all_bboxes = np.concatenate(all_bboxes, axis=0)
    all_scores = np.concatenate(all_scores, axis=0)
    pred_bboxes = []
    pred_scores = []
    pred_clsid = []
    mx.nd.waitall()
    from utils.nms.nms import gpu_nms_wrapper
    nms_wrapper = gpu_nms_wrapper(thresh=.3, device_id=7)
    for j in range(1, all_scores.shape[1]):
        cls_scores = all_scores[:, j, np.newaxis]
        cls_boxes = all_bboxes[:, 4:8] if class_agnostic else all_bboxes[:, j * 4:(j + 1) * 4]
        cls_dets = np.hstack((cls_boxes, cls_scores))
        # cls_dets = nd.contrib.box_nms(nd.array(cls_dets, ctx=mx.cpu(ctx[0].device_id)),
        #                               overlap_thresh=nms_threshold, coord_start=0, score_index=4, id_index=-1,
        #                               force_suppress=True, in_format='corner',
        #                               out_format='corner').asnumpy()
        keep = nms_wrapper(cls_dets.astype('f'))
        cls_dets = cls_dets[keep]
        cls_dets = cls_dets[cls_dets[:, -1] > threshold, :]
        pred_bboxes.append(cls_dets[:, :4])
        pred_scores.append(cls_dets[:, 4])
        pred_clsid.append(j * np.ones(shape=(cls_dets.shape[0],), dtype=np.int))
    pred_bboxes = np.concatenate(pred_bboxes, axis=0)
    pred_scores = np.concatenate(pred_scores, axis=0)
    pred_clsid = np.concatenate(pred_clsid, axis=0)
    if viz:
        import gluoncv
        import matplotlib.pyplot as plt
        gluoncv.utils.viz.plot_bbox(img_ori, bboxes=pred_bboxes, scores=pred_scores, labels=pred_clsid,
                                    thresh=.5)
        plt.show()
    return pred_bboxes, pred_scores, pred_clsid
