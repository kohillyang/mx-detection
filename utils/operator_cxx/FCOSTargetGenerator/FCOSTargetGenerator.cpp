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
MOBULA_KERNEL fcos_target_gen_kernel(const int feature_h, const int feature_w, const int feature_ch, const int stride,
const T* bboxes, const int number_of_bboxes, T distance_min, T distance_max, T* output) {
    UNUSED(feature_h);
    UNUSED(feature_w);
    UNUSED(stride);
    UNUSED(bboxes);
    UNUSED(number_of_bboxes);
    UNUSED(output);

    for(int f_w=0; f_w < feature_w; f_w++){
        for(int f_h=0; f_h < feature_h; f_h ++){
            T ori_x = f_w * stride + static_cast<T>(stride) / 2;
            T ori_y = f_h * stride + static_cast<T>(stride) / 2;
            int bbox_target=-1;
            T gt_bbox_min_area = 1e99;
            T *output_base = output + f_h * feature_w * feature_ch + f_w * feature_ch;
            for(int n_b=0; n_b< number_of_bboxes; n_b++){
                const T *p_bboxes_offset = bboxes + n_b * 5;
                T x0 = p_bboxes_offset[0];
                T y0 = p_bboxes_offset[1];
                T x1 = p_bboxes_offset[2];
                T y1 = p_bboxes_offset[3];
                if(ori_x >= x0 && ori_x <= x1 && ori_y >= y0 && ori_y <= y1){
                    T delta_l = ori_x - x0;
                    T delta_t = ori_y - y0;
                    T delta_r = x1 - ori_x;
                    T delta_b = y1 - ori_y;
                    T gt_area = (x1 - x0) * (y1 - y0);
                    T max_delta = std::max(std::max(delta_l, delta_t), std::max(delta_r, delta_b));
                    // If a location, even with multi-level prediction used, is still assigned to more
                    // than one ground-truth boxes, we simply choose the groundtruth box with minimal area as its target
                    if(max_delta >=distance_min && max_delta <= distance_max){
                        if (bbox_target < 0 || gt_area < gt_bbox_min_area){
                            bbox_target = n_b;
                        }
                    }
                }
            }
            if(bbox_target >= 0){ // positive sample
                const T *p_bboxes_offset = bboxes + bbox_target * 5;
                T x0 = p_bboxes_offset[0];
                T y0 = p_bboxes_offset[1];
                T x1 = p_bboxes_offset[2];
                T y1 = p_bboxes_offset[3];
                int class_id = static_cast<int>(p_bboxes_offset[4]);

                T delta_l = ori_x - x0;
                T delta_t = ori_y - y0;
                T delta_r = x1 - ori_x;
                T delta_b = y1 - ori_y;

                output_base[0] = 1; // one channel for mask.
                output_base[1] = delta_l;
                output_base[2] = delta_t;
                output_base[3] = delta_r;
                output_base[4] = delta_b;
                T center_ness = std::min(delta_r, delta_l) * std::min(delta_t, delta_b);
                center_ness /= std::max(delta_r, delta_l) * std::max(delta_t, delta_b) + 1e-1;
                center_ness = std::sqrt(center_ness);
                output_base[5] = center_ness;
                assert(6 + class_id -1 < feature_ch);
                output_base[6 + class_id  - 1] = 1; // class_id starts with 1
            }else{
            }
        }
    }

} // fcos_target_gen_kernel


}  // namespace mobula
