/*
 * author: kohill
 */
#include "bilinear.h"
#include "mobula_op.h"
#include <memory>
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <cmath>
namespace mobula {

#define UNUSED(expr) do { (void)(expr); } while (0)

template <typename T>
MOBULA_KERNEL retinanet_target_gen_kernel(const int image_h, const int image_w,
const int feature_h, const int feature_w, const int feature_ch, const int stride,
const T* bboxes, const int number_of_bboxes, const T negative_iou_threshold, const T positive_iou_threshold,
T* anchors_base_wh, const int anchors_base_wh_size, T* output) {

    for(int f_w=0; f_w < feature_w; f_w++){
        for(int f_h=0; f_h < feature_h; f_h ++){
            T ori_x = f_w * stride + static_cast<T>(stride) / 2;
            T ori_y = f_h * stride + static_cast<T>(stride) / 2;
            for(int anchor_idx=0; anchor_idx<anchors_base_wh_size; ++anchor_idx){
                T *output_base = output + f_h * feature_w * anchors_base_wh_size * feature_ch;
                output_base += f_w * anchors_base_wh_size * feature_ch;
                output_base += anchor_idx * feature_ch;

                T anchor_w = anchors_base_wh[anchor_idx * 2 + 0];
                T anchor_h = anchors_base_wh[anchor_idx * 2 + 1];
                T anchor_x0 = ori_x - anchor_w / 2;
                T anchor_y0 = ori_y - anchor_h / 2;
                T anchor_x1 = ori_x + anchor_w / 2;
                T anchor_y1 = ori_y + anchor_h / 2;
                if(anchor_x0 < 0){
                    anchor_x0 = 0;
                }
                if(anchor_y0 < 0){
                    anchor_y0 = 0;
                }
                if(anchor_x1 > image_w){
                    anchor_x1 = image_w;
                }
                if(anchor_y1 > image_h){
                    anchor_y1 = image_h;
                }
                if(anchor_x0 >= anchor_x1 || anchor_y0 >= anchor_y1){
                    continue;
                }
                // If the maximum IoU between this anchor and the gt_boxes is greater than a threshold,
                // then it will be assigned as positive.
                // If the maximum IoU between this anchor and the gt_boxes is less than a threshold,
                // then it will be assigned as negative.
            }
        }
    }

} // fcos_target_gen_kernel


}  // namespace mobula
