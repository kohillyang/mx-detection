/*
 * author: kohill
 */
#include "mobula_op.h"
#include <iostream>
namespace mobula {

#define UNUSED(expr) do { (void)(expr); } while (0)

template <typename T>
MOBULA_FUNC void fcos_target_regression(const T *prediction, int feature_n, int feature_h, int feature_w,  int feature_ch, int stride, T* output) {
    UNUSED(prediction);
    UNUSED(feature_h);
    UNUSED(feature_w);
    UNUSED(feature_ch);
    UNUSED(stride);
    UNUSED(output);
    UNUSED(feature_n);

    const int ch_l=0, ch_t=1, ch_r=2, ch_b=3;
    int ch_center_ness = 4;
    int ch_cls_start = 5;
    int ch_cls_end = feature_ch;

    int n_bbox = 0;

    for(int f_w=0; f_w < feature_w; f_w++){
        for(int f_h=0; f_h < feature_h; f_h ++){
            T ori_x = f_w * stride + static_cast<T>(stride) / 2;
            T ori_y = f_h * stride + static_cast<T>(stride) / 2; 
            T delta_l = *(prediction + ch_l * feature_h * feature_w + f_h * feature_w + f_w);
            T delta_t = *(prediction + ch_t * feature_h * feature_w + f_h * feature_w + f_w);
            T delta_r = *(prediction + ch_r * feature_h * feature_w + f_h * feature_w + f_w);
            T delta_b = *(prediction + ch_b * feature_h * feature_w + f_h * feature_w + f_w);
            T pred_x0 = ori_x - delta_l;
            T pred_y0 = ori_y - delta_t;
            T pred_x1 = ori_x + delta_r;
            T pred_y1 = ori_y + delta_b;
            T centerness_score = *(prediction + ch_center_ness * feature_h * feature_w + f_h * feature_w + f_w);

            for(int class_id=ch_cls_start; class_id<ch_cls_end; class_id++){
                T class_score = *(prediction + class_id * feature_h * feature_w + f_h * feature_w + f_w);
                T score_used_for_ranking = centerness_score * class_score;
//                std::cout << class_score << " " << centerness_score << std::endl;
                UNUSED(score_used_for_ranking);
                output[n_bbox*6 + 0] = pred_x0;             
                output[n_bbox*6 + 1] = pred_y0;             
                output[n_bbox*6 + 2] = pred_x1;             
                output[n_bbox*6 + 3] = pred_y1;             
                output[n_bbox*6 + 4] = score_used_for_ranking;
                output[n_bbox*6 + 5] = class_id - ch_cls_start;
                n_bbox ++;      
            } 
        }
    }

} // fcos_target_gen_kernel


}  // namespace mobula
