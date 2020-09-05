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
namespace mobula {

#define UNUSED(expr) do { (void)(expr); } while (0)

template <typename T>
MOBULA_FUNC void retinanet_regression(
		const int image_h,
		const int image_w,
		const int n_batch, const int feature_h,
		const int feature_w,
		const int number_of_classes,
		T* pointer_reg_preds,
		T* pointer_cls_preds ,
		const int stride,
		const T* anchors_base_wh,
		const int anchors_base_wh_size,
		const T cls_threshold, T* output) {
	int n_bbox=0;
	Tensor5D<T> tensor_cls_preds = Tensor5D<T>(pointer_cls_preds, n_batch, number_of_classes, anchors_base_wh_size, feature_h, feature_w);
	Tensor5D<T> tensor_reg_preds = Tensor5D<T>(pointer_reg_preds, n_batch,  4,  anchors_base_wh_size, feature_h, feature_w);
	for (int n_image=0; n_image < n_batch; ++n_image){
		int nbbox = 0;
		T *bbox_imgae_base = output + n_image * feature_h * feature_w * anchors_base_wh_size * 6;
	    for(int f_w=0; f_w < feature_w; f_w++){
	        for(int f_h=0; f_h < feature_h; f_h ++){
	            T ori_x = f_w * stride + static_cast<T>(stride) / 2;
	            T ori_y = f_h * stride + static_cast<T>(stride) / 2;
	            for(int anchor_idx=0; anchor_idx<anchors_base_wh_size; ++anchor_idx){

	                T anchor_w = anchors_base_wh[anchor_idx * 2 + 0];
	                T anchor_h = anchors_base_wh[anchor_idx * 2 + 1];

	                T anchor_x0 = ori_x - anchor_w / 2;
	                T anchor_y0 = ori_y - anchor_h / 2;
	                T anchor_x1 = ori_x + anchor_w / 2;
	                T anchor_y1 = ori_y + anchor_h / 2;
	                if(anchor_x0 < 0){
						anchor_x0 = 0;
						continue;
					}
	                if(anchor_y0 < 0){
						anchor_y0 = 0;
                        continue;
	                		}
	                if(anchor_x1 > image_w){
	                			anchor_x1 = image_w;
                        continue;
					}
	                if(anchor_y1 > image_h){
	                			anchor_y1 = image_h;
						continue;
					}
	                if(anchor_x0 >= anchor_x1 || anchor_y0 >= anchor_y1){
	                    continue;
	                		}
	                T net_pred_0 = tensor_reg_preds(0, 0, anchor_idx, f_h, f_w);
	                T net_pred_1 = tensor_reg_preds(0, 1, anchor_idx, f_h, f_w);
	                T net_pred_2 = tensor_reg_preds(0, 2, anchor_idx, f_h, f_w);
	                T net_pred_3 = tensor_reg_preds(0, 3, anchor_idx, f_h, f_w);

	                T pred_x0 = net_pred_0 * (anchor_x1 - anchor_x0 + 1) + anchor_x0;
	                T pred_y0 =net_pred_1 * (anchor_y1 - anchor_y0 + 1) + anchor_y0;
	                T pred_w = std::exp(net_pred_2 + std::log(anchor_x1 - anchor_x0));
	                T pred_h = std::exp(net_pred_3+ std::log(anchor_y1 - anchor_y0));

	                for(int nc=0; nc< number_of_classes; nc++){
						T score = tensor_cls_preds(0, nc, anchor_idx, f_h, f_w);
	                	if(score > cls_threshold){
							bbox_imgae_base[nbbox * 6 + 0] = pred_x0;
							bbox_imgae_base[nbbox * 6 + 1] = pred_y0;
							bbox_imgae_base[nbbox * 6 + 2] = pred_x0 + pred_w - 1;
							bbox_imgae_base[nbbox * 6 + 3] = pred_y0 + pred_h - 1;
							bbox_imgae_base[nbbox * 6 + 4] = score;
							bbox_imgae_base[nbbox * 6 + 5] = nc;
							nbbox ++;
	                		}
	                }
	            }
	        }
	    }
	}


} // fcos_target_gen_kernel


}  // namespace mobula
