/*
 * author: kohill
 */
#include "mobula_op.h"
#include <iostream>
#include "../tensor.hpp"
namespace mobula {

#define UNUSED(expr) do { (void)(expr); } while (0)

template <typename T>
MOBULA_KERNEL fcos_target_regression_kernel(
        int outsize,
		T* pointer_loc_pred,
		T* pointer_cls_pred,
		int nbatch,
		int number_of_classes_no_background,
		int feature_h,
		int feature_w,
		int stride,
		int image_width,
		int image_height,
		T* output) {

    UNUSED(outsize);
	parfor(nbatch * feature_h * feature_w, [&](int index){
		int batch_idx = index / (feature_h * feature_w);
		index -= batch_idx * (feature_h * feature_w);
		int h_idx = index / feature_w;
		index -= h_idx * feature_w;
		int w_idx = index;
        T ori_x = w_idx * stride + static_cast<T>(stride) / 2;
        T ori_y = h_idx * stride + static_cast<T>(stride) / 2;
    	Tensor4D<T> tensor_loc_pred = Tensor4D<T>(pointer_loc_pred, nbatch, 5, feature_h, feature_w);
    	Tensor4D<T> tensor_cls_pred = Tensor4D<T>(pointer_cls_pred, nbatch, number_of_classes_no_background, feature_h, feature_w);
    	Tensor5D<T> tensor_output = Tensor5D<T>(output, nbatch, number_of_classes_no_background, feature_h, feature_w, 6);

        T delta_l = 		 tensor_loc_pred(batch_idx, 0, h_idx, w_idx);
        T delta_t = 		 tensor_loc_pred(batch_idx, 1, h_idx, w_idx);
        T delta_r = 		 tensor_loc_pred(batch_idx, 2, h_idx, w_idx);
        T delta_b =		     tensor_loc_pred(batch_idx, 3, h_idx, w_idx);
        T centerness_score = tensor_loc_pred(batch_idx, 4, h_idx, w_idx);

        T pred_x0 = ori_x - delta_l;
        T pred_y0 = ori_y - delta_t;
        T pred_x1 = ori_x + delta_r;
        T pred_y1 = ori_y + delta_b;
        if(pred_x0 <0){
            pred_x0 = 0;
        }
        if(pred_y0 <0){
            pred_y0 = 0;
        }
        if(pred_x1 <0){
            pred_x1 = 0;
        }
        if(pred_y1 <0){
            pred_y1 = 0;
        }

        if(pred_x0 > image_width - 1){
            pred_x0 = image_width - 1;
        }
        if(pred_y0 > image_height - 1){
            pred_y0 = image_height - 1;
        }
        if(pred_x1 >image_width - 1){
            pred_x1 = image_width - 1;
        }
        if(pred_y1 > image_height - 1){
            pred_y1 = image_height - 1;
        }

        for(int class_id=0; class_id<number_of_classes_no_background; class_id++){
            T class_score = tensor_cls_pred(batch_idx, class_id, h_idx, w_idx);
            T score_used_for_ranking = centerness_score * class_score;
            tensor_output(batch_idx, class_id, h_idx, w_idx, 0) = pred_x0;
            tensor_output(batch_idx, class_id, h_idx, w_idx, 1) = pred_y0;
            tensor_output(batch_idx, class_id, h_idx, w_idx, 2) = pred_x1;
            tensor_output(batch_idx, class_id, h_idx, w_idx, 3) = pred_y1;
            if(pred_y1 >= pred_y0 && pred_x1 >= pred_x0){
                tensor_output(batch_idx, class_id, h_idx, w_idx, 4) = score_used_for_ranking;
            }else{
                tensor_output(batch_idx, class_id, h_idx, w_idx, 4) = 0;
            }
            tensor_output(batch_idx, class_id, h_idx, w_idx, 5) = class_id;
        }
	});
} // fcos_target_regression


}  // namespace mobula
