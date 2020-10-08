/*
 * author: kohill
 */
#include "mobula_op.h"
#if USING_CUDA
#include <cuda.h>
#else
#include <algorithm>
#include <cmath>
using std::exp;
using std::log;
using std::max;
using std::min;
using std::pow;
#endif  // USING_CUDA

#include "../tensor.hpp"
namespace mobula {

template <typename T>
MOBULA_DEVICE T box_iou(T x0, T y0, T x1, T y1, T hat_x0, T hat_y0, T hat_x1, T hat_y1){
    T i_w = min(hat_x1, x1) - max(x0, hat_x0);
    T i_h = min(hat_y1, y1) - max(y0, hat_y0);
    if(i_w >0 && i_h >0){
        return i_w * i_h / ((x1 - x0) * (y1 - y0) + (hat_x1 - hat_x0) * (hat_y1 - hat_y0) - i_w * i_h);
    }else{
        return 0;
    }
}
#define UNUSED(expr) do { (void)(expr); } while (0)

template <typename T>
MOBULA_KERNEL retinanet_target_gen_kernel(
        int threads_ref_number,
        int n_batch,
        int image_h,
        int image_w,
        int feature_h,
        int feature_w,
        int num_classes,
        int stride,
        T *pointer_bboxes,
        int number_of_bboxes,
        T negative_iou_threshold,
        T positive_iou_threshold,
        T *anchors_base_wh,
		T *bbox_norm_coef,
        int anchors_base_wh_size,
        T *loc_targets_output,
        T *cls_targets_output,
		T *regmask_targets_output,
		T *clsmask_targets_output) {
	parfor(n_batch*feature_h*feature_w, [&](int index) {
		int batch_idx = index / (feature_h*feature_w);
		index -= batch_idx * (feature_h*feature_w);
		int h_idx = index / feature_w;
		index -= h_idx * feature_w;
		int w_idx = index;

		T ori_x = w_idx * stride + static_cast<T>(stride) / 2;
		T ori_y = h_idx * stride + static_cast<T>(stride) / 2;
		Tensor3D<T> tensor_bboxes = Tensor3D<T>(pointer_bboxes, n_batch, number_of_bboxes, 5);
		// Determine the real number of bboxes
		int real_number_of_bboxes = 0;
		for(int i=0; i<number_of_bboxes; ++i){
			T class_id = tensor_bboxes(batch_idx, i, 4);
			if(class_id >=0){
				real_number_of_bboxes += 1;
			}else{
			    break;
			}
		}
		Tensor4D<T> tensor_loc_targets_output = Tensor4D<T>(loc_targets_output, n_batch, anchors_base_wh_size *4, feature_h, feature_w);
		// num_classes should not include the background class.
		Tensor4D<T> tensor_cls_targets_output = Tensor4D<T>(cls_targets_output, n_batch, anchors_base_wh_size *num_classes, feature_h, feature_w);
		Tensor4D<T> tensor_regmask_targets_output = Tensor4D<T>(regmask_targets_output, n_batch, anchors_base_wh_size *4, feature_h, feature_w);
		Tensor4D<T> tensor_clsmask_targets_output = Tensor4D<T>(clsmask_targets_output, n_batch, anchors_base_wh_size *num_classes, feature_h, feature_w);

		for(int anchor_idx=0; anchor_idx<anchors_base_wh_size; ++anchor_idx){

			T anchor_w = anchors_base_wh[anchor_idx * 2 + 0];
			T anchor_h = anchors_base_wh[anchor_idx * 2 + 1];

			T anchor_x0 = ori_x - anchor_w / 2;
			T anchor_y0 = ori_y - anchor_h / 2;
			T anchor_x1 = ori_x + anchor_w / 2;
			T anchor_y1 = ori_y + anchor_h / 2;

//            // clip anchors which are out of bounds.
//			anchor_x0 = max(static_cast<T>(0), anchor_x0);
//			anchor_y0 = max(static_cast<T>(0), anchor_y0);
//			anchor_x1 = max(static_cast<T>(0), anchor_x1);
//			anchor_y1 = max(static_cast<T>(0), anchor_y1);
//			anchor_x0 = min(static_cast<T>(image_w), anchor_x0);
//			anchor_y0 = min(static_cast<T>(image_h), anchor_y0);
//			anchor_x1 = min(static_cast<T>(image_w), anchor_x1);
//			anchor_y1 = min(static_cast<T>(image_h), anchor_y1);
            if(anchor_x0 >= anchor_x1 || anchor_y0 >= anchor_y1){
                continue;
            }
			// If the maximum IoU between this anchor and the gt_boxes is greater than a threshold,
			// then it will be assigned as positive.
			// If the maximum IoU between this anchor and the gt_boxes is less than a threshold,
			// then it will be assigned as negative.
			T max_iou = 0;
			int gt_bbox_idx_with_max_iou = -1;
			for(int gt_bbox_idx=0; gt_bbox_idx < real_number_of_bboxes; gt_bbox_idx++){
				T gt_x0 = tensor_bboxes(batch_idx, gt_bbox_idx, 0);
				T gt_y0 = tensor_bboxes(batch_idx, gt_bbox_idx, 1);
				T gt_x1 = tensor_bboxes(batch_idx, gt_bbox_idx, 2);
				T gt_y1 = tensor_bboxes(batch_idx, gt_bbox_idx, 3);
				T iou = box_iou(anchor_x0, anchor_y0, anchor_x1, anchor_y1, gt_x0, gt_y0, gt_x1, gt_y1);
				if (iou > max_iou){
					max_iou = iou;
					gt_bbox_idx_with_max_iou = gt_bbox_idx;
				}
			}
			if(gt_bbox_idx_with_max_iou >= 0 && max_iou >positive_iou_threshold){
				// positive sample
				T gt_x0 = tensor_bboxes(batch_idx, gt_bbox_idx_with_max_iou, 0);
				T gt_y0 = tensor_bboxes(batch_idx, gt_bbox_idx_with_max_iou, 1);
				T gt_x1 = tensor_bboxes(batch_idx, gt_bbox_idx_with_max_iou, 2);
				T gt_y1 = tensor_bboxes(batch_idx, gt_bbox_idx_with_max_iou, 3);
				T class_id_T = tensor_bboxes(batch_idx, gt_bbox_idx_with_max_iou, 4);
				int class_id = static_cast<int>(class_id_T);

                assert(gt_x1 - gt_x0 > 1e-3);
                assert(gt_y1 - gt_y0 > 1e-3);

				tensor_loc_targets_output(batch_idx, anchor_idx * 4 + 0, h_idx, w_idx) = (gt_x0 - anchor_x0) / (anchor_x1 - anchor_x0 + 1) / bbox_norm_coef[0];
				tensor_loc_targets_output(batch_idx, anchor_idx * 4 + 1, h_idx, w_idx) = (gt_y0 - anchor_y0) / (anchor_y1 - anchor_y0 + 1) / bbox_norm_coef[1];
				tensor_loc_targets_output(batch_idx, anchor_idx * 4 + 2, h_idx, w_idx) = (log(gt_x1 - gt_x0) - log(anchor_x1 - anchor_x0)) / bbox_norm_coef[2];
				tensor_loc_targets_output(batch_idx, anchor_idx * 4 + 3, h_idx, w_idx) = (log(gt_y1 - gt_y0) - log(anchor_y1 - anchor_y0)) / bbox_norm_coef[3];
				tensor_cls_targets_output(batch_idx, anchor_idx * num_classes + class_id, h_idx, w_idx) = 1;
				for(int mask_cls_idx=0; mask_cls_idx<num_classes; mask_cls_idx++){
					tensor_clsmask_targets_output(batch_idx, anchor_idx * num_classes + mask_cls_idx, h_idx, w_idx) = 1;
				}
				tensor_regmask_targets_output(batch_idx, anchor_idx * 4 + 0, h_idx, w_idx) = 1;
				tensor_regmask_targets_output(batch_idx, anchor_idx * 4 + 1, h_idx, w_idx) = 1;
				tensor_regmask_targets_output(batch_idx, anchor_idx * 4 + 2, h_idx, w_idx) = 1;
				tensor_regmask_targets_output(batch_idx, anchor_idx * 4 + 3, h_idx, w_idx) = 1;

			} else if(gt_bbox_idx_with_max_iou >= 0 && max_iou >negative_iou_threshold){
				// just ignore these sample whose ious are between negative_iou_threshold and positive_iou_threshold.
				// we assume the default value is zero, so nothing need to do here.
			} else{
				// negative sample.
				// cls mask should be 1, label should be 0;
				for(int mask_cls_idx=0; mask_cls_idx<num_classes; mask_cls_idx++){
					tensor_clsmask_targets_output(batch_idx, anchor_idx * num_classes + mask_cls_idx, h_idx, w_idx) = 1;
				}
				// reg mask should be zero.
				tensor_regmask_targets_output(batch_idx, anchor_idx * 4 + 0, h_idx, w_idx) = 0;
				tensor_regmask_targets_output(batch_idx, anchor_idx * 4 + 1, h_idx, w_idx) = 0;
				tensor_regmask_targets_output(batch_idx, anchor_idx * 4 + 2, h_idx, w_idx) = 0;
				tensor_regmask_targets_output(batch_idx, anchor_idx * 4 + 3, h_idx, w_idx) = 0;
			}
		}
	});

} // fcos_target_gen_kernel


}  // namespace mobula
