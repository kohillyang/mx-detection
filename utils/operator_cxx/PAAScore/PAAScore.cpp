/*
 * author: kohill
 */
#include "mobula_op.h"
#include <memory>
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <cmath>
#include "../tensor.hpp"
//#include <iostream>
//#include <cstdio>
//using namespace std;

namespace mobula {

#define UNUSED(expr) do { (void)(expr); } while (0)
template <typename T>
static MOBULA_DEVICE T box_iou(T x0, T y0, T x1, T y1, T hat_x0, T hat_y0, T hat_x1, T hat_y1){
    T i_w = min(hat_x1, x1) - max(x0, hat_x0);
    T i_h = min(hat_y1, y1) - max(y0, hat_y0);
    if(i_w >0 && i_h >0){
        return i_w * i_h / ((x1 - x0) * (y1 - y0) + (hat_x1 - hat_x0) * (hat_y1 - hat_y0) - i_w * i_h);
    }else{
        return 0;
    }
}
template <typename T>
MOBULA_KERNEL  paa_score_kernel(
		const int stride,
		const int image_h,
		const int image_w,
		const int nbatch,
		const int feature_h,
		const int feature_w,
		const int number_of_anchors,
		const int number_of_classes,
		const int gt_boxes_padded_length,
		T* pointer_reg_preds,
		T* pointer_cls_preds,
		T *pointer_gt_boxes,
		T *pointer_gt_boxes_number,
		T *pointer_anchors_base_wh,
		T* output) {
	// nbatch, number_of_anchors_times_number_of_classes, feature_h, feature_w
	parfor(nbatch * number_of_anchors * feature_h * feature_w, [&](int index) {
		Tensor5D<T> tensor_reg_preds = Tensor5D<T>(pointer_reg_preds, nbatch, number_of_anchors, 4, feature_h, feature_w);
		Tensor5D<T> tensor_cls_preds = Tensor5D<T>(pointer_cls_preds, nbatch, number_of_anchors, number_of_classes, feature_h, feature_w);
		Tensor3D<T> tensor_gt_boxes = Tensor3D<T>(pointer_gt_boxes, nbatch, gt_boxes_padded_length, 5);
		Tensor5D<T> tensor_output = Tensor5D<T>(output, nbatch, number_of_anchors, number_of_classes, feature_h, feature_w);
		int total_idx = index;
		int batch_idx = total_idx / (number_of_anchors * feature_h * feature_w);
		total_idx -= batch_idx * (number_of_anchors * feature_h * feature_w);

		int anchor_idx = total_idx /  (feature_h * feature_w);
		total_idx -= anchor_idx *  (feature_h * feature_w);

		int h_idx = total_idx / feature_w;
		total_idx -= h_idx * feature_w;

		int w_idx = total_idx;

		int gt_boxes_number = static_cast<int>(pointer_gt_boxes_number[batch_idx]);

		// for each anchor on a position, we have several bboxes, which share same bbox width and height,
        T anchor_w = pointer_anchors_base_wh[anchor_idx * 2 + 0];
        T anchor_h = pointer_anchors_base_wh[anchor_idx * 2 + 1];
        T ori_x = w_idx * stride + static_cast<T>(stride) / 2;
        T ori_y = h_idx * stride + static_cast<T>(stride) / 2;

        T anchor_x0 = ori_x - anchor_w / 2;
        T anchor_y0 = ori_y - anchor_h / 2;
        T anchor_x1 = ori_x + anchor_w / 2;
        T anchor_y1 = ori_y + anchor_h / 2;
        if(anchor_x0 < 0 || anchor_y0 < 0 || anchor_x1 > image_w || anchor_y1 > image_h){
        	return;
        	}
        T net_pred_0 = tensor_reg_preds(batch_idx, anchor_idx, 0, h_idx, w_idx);
        T net_pred_1 = tensor_reg_preds(batch_idx, anchor_idx, 1, h_idx, w_idx);
        T net_pred_2 = tensor_reg_preds(batch_idx, anchor_idx, 2, h_idx, w_idx);
        T net_pred_3 = tensor_reg_preds(batch_idx, anchor_idx, 3, h_idx, w_idx);

        T pred_x0 = net_pred_0 * (anchor_x1 - anchor_x0 + 1) + anchor_x0;
        T pred_y0 = net_pred_1 * (anchor_y1 - anchor_y0 + 1) + anchor_y0;
        T pred_w = exp(net_pred_2 + log(anchor_x1 - anchor_x0));
        T pred_h = exp(net_pred_3 + log(anchor_y1 - anchor_y0));
        	// There maybe more than one gt_boxes with the same class, we choose the one with the max iou.
        for(int class_id=0; class_id < number_of_classes; class_id++){
        	T max_iou = 0;
        	int max_iou_gt_box_idx = -1;
            for(int gt_box_idx=0; gt_box_idx< gt_boxes_number; gt_box_idx++){
            	T gt_x0 = tensor_gt_boxes(batch_idx, gt_box_idx, 0) * .1;
            	T gt_y0 = tensor_gt_boxes(batch_idx, gt_box_idx, 1) * .1;
            	T gt_x1 = tensor_gt_boxes(batch_idx, gt_box_idx, 2) * .2;
            	T gt_y1 = tensor_gt_boxes(batch_idx, gt_box_idx, 3) * .2;
            	T gt_class_id =  tensor_gt_boxes(batch_idx, gt_box_idx, 4);
				if(gt_class_id == class_id){
					T iou = box_iou(pred_x0, pred_y0, pred_w, pred_h, gt_x0, gt_y0, gt_x1, gt_y1);
					if(iou >= max_iou){
						max_iou = iou;
						max_iou_gt_box_idx = gt_box_idx;
					}
				}
            		}
            if(max_iou_gt_box_idx >= 0 ){
            	int gt_box_idx = max_iou_gt_box_idx;
            	T gt_x0 = tensor_gt_boxes(batch_idx, gt_box_idx, 0) * .1;
            	T gt_y0 = tensor_gt_boxes(batch_idx, gt_box_idx, 1) * .1;
            	T gt_x1 = tensor_gt_boxes(batch_idx, gt_box_idx, 2) * .2;
            	T gt_y1 = tensor_gt_boxes(batch_idx, gt_box_idx, 3) * .2;
            	T gt_class_id =  tensor_gt_boxes(batch_idx, gt_box_idx, 4);
            	T iou = max_iou;
            	T net_score_pred =  tensor_cls_preds(batch_idx, anchor_idx, gt_class_id, h_idx, w_idx);
            	T score = net_score_pred ;
            			tensor_output(batch_idx, anchor_idx, class_id, h_idx, w_idx) = score;;
			}
        }
	}); // parfor

} // paa_score_kernel


}  // namespace mobula
