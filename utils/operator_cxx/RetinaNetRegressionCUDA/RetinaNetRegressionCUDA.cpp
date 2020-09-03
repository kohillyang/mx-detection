/*
 * author: kohill
 */
#include "mobula_op.h"
#include <memory>
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <cmath>
namespace mobula {

#define UNUSED(expr) do { (void)(expr); } while (0)

template <typename T>
MOBULA_FUNC void retinanet_regression(const int image_h, const int image_w,
		const int n_batch, const int feature_h,
		const int feature_w, const int n_anchor, const int n_ch, const T* feature,
		const int stride, const T* anchors_base_wh, const int anchors_base_wh_size,
		const T cls_threshold, T* output) {
	int n_bbox=0;
	assert(anchors_base_wh_size==n_anchor);
	int number_of_classes = n_ch - 4; // 4 reg, and others are for classification.
	for (int n_image=0; n_image < n_batch; ++n_image){
		T * bbox_imgae_base = output + n_image * feature_h * feature_w * n_anchor * 6;
		const T * feature_imgae_base = feature + n_image * feature_h * feature_w * n_anchor * n_ch;
		int nbbox = 0;
	    for(int f_w=0; f_w < feature_w; f_w++){
	        for(int f_h=0; f_h < feature_h; f_h ++){
	            T ori_x = f_w * stride + static_cast<T>(stride) / 2;
	            T ori_y = f_h * stride + static_cast<T>(stride) / 2;
	            for(int anchor_idx=0; anchor_idx<anchors_base_wh_size; ++anchor_idx){
	            	const T *feature_base = feature_imgae_base + f_h * feature_w * n_anchor * n_ch;
	            	feature_base += f_w * n_anchor * n_ch;
	            	feature_base += anchor_idx * n_ch;

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
	                T pred_x0 = feature_base[0] * (anchor_x1 - anchor_x0 + 1) + anchor_x0;
	                T pred_y0 = feature_base[1] * (anchor_y1 - anchor_y0 + 1) + anchor_y0;
	                T pred_w = std::exp(feature_base[2] + std::log(anchor_x1 - anchor_x0));
	                T pred_h = std::exp(feature_base[3] + std::log(anchor_y1 - anchor_y0));

	                for(int nc=0; nc< number_of_classes; nc++){
	                	if(feature_base[4 + nc] > cls_threshold){
	                		bbox_imgae_base[nbbox * 6 + 0] = pred_x0;
	                		bbox_imgae_base[nbbox * 6 + 1] = pred_y0;
	                		bbox_imgae_base[nbbox * 6 + 2] = pred_x0 + pred_w - 1;
	                		bbox_imgae_base[nbbox * 6 + 3] = pred_y0 + pred_h - 1;
	                		bbox_imgae_base[nbbox * 6 + 4] = feature_base[4 + nc];
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
